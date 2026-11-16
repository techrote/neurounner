"""GAL-011 host authority, observation boundaries, local policy regression."""
import json
import tempfile
from pathlib import Path
import unittest
from lab import START, HostRefusal, sha
from material_corridors import CorridorWorld
from frontier_memory import (Ledger, EVAL_BASE, Scorer, ARMS_TEST, choose_teacher,
                             features, train, one_episode, mask_memory_features,
                             MEMORY_INDICES, run, audit)

class FrontierMemoryTests(unittest.TestCase):
    def test_no_invisible_cell_appears_in_ledger(self):
        w=CorridorWorld(EVAL_BASE,'tree','lift','test');ledger=Ledger()
        crop=ledger.observe(w)
        self.assertEqual(ledger.known,crop)
        self.assertTrue(all(abs(x-START[0])+abs(y-START[1])<=4 for x,y in ledger.known))
        self.assertLess(len(ledger.known),len(w.corridor)+30)

    def test_teacher_does_not_require_goal_or_corridor_field(self):
        w=CorridorWorld(EVAL_BASE,'tree','lift','test');ledger=Ledger()
        ledger.observe(w);menu=w.legal()
        class GoalDenied:
            @property
            def cursor(self):return w.cursor
        action=choose_teacher(GoalDenied(),ledger,menu)
        self.assertIn(action,menu)

    def test_route_follows_one_step_before_target(self):
        w=CorridorWorld(EVAL_BASE,'tree','lift','test');ledger=Ledger()
        ledger.observe(w)
        ledger.known={(8,24):58,(9,24):58,(10,24):0}
        w.cursor=(8,24)
        menu={n:p for n,p in w.legal().items() if n=='STOP' or n.startswith('LIFT_')}
        self.assertEqual(ledger.route(w.cursor),((9,24),(10,24)))
        self.assertEqual(choose_teacher(w,ledger,menu),'LIFT_E_1')

    def test_host_rejects_stale_lease(self):
        w=CorridorWorld(EVAL_BASE,'tree','lift','test')
        a,t=w.offer();chosen=next(n for n in a if n.startswith('INK_'))
        w.apply(chosen,t,original_response='test')
        with self.assertRaises(HostRefusal):w.apply(chosen,t,original_response='stale')

    def test_teacher_completes_independent_branch_seeds(self):
        for i in range(5):
            w,row,_=one_episode(Scorer(),'lift','tree',EVAL_BASE+i,mode='teacher')
            self.assertEqual(row['correct'],32)
            self.assertFalse(w.own&w.protected)
            self.assertLess(row['decisions'],96)
            self.assertGreater(row['lift_decisions'],0)

    def test_memory_knockout_keeps_shape_and_preserves_local_sensor(self):
        w=CorridorWorld(EVAL_BASE,'path','lift','test');l=Ledger();l.observe(w)
        a,p=next((a,p) for a,p in w.legal().items() if a.startswith('INK_'))
        v=features(w,l,a,p,l.route(w.cursor));masked=mask_memory_features(v)
        self.assertEqual(len(masked),len(v))
        self.assertEqual(masked[0],v[0]);self.assertEqual(masked[1],v[1])
        self.assertEqual(masked[17],v[17])
        for i in MEMORY_INDICES:self.assertEqual(masked[i],0.)

    def test_trained_weights_are_nonzero(self):
        m,s=train('lift',4)
        self.assertGreater(s['updates'],0)
        self.assertTrue(any(abs(v)>1e-8 for v in m.weights))

    def test_miniature_campaign_and_full_source_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=run(tmp,4,4,0)
            self.assertEqual(len(m['checkpoints_sha256']),2)
            outcome=audit(tmp)
            self.assertEqual(outcome['checkpoints_retrained'],2)
            self.assertEqual(outcome['heldout_episodes_replayed'],32)
            self.assertGreater(outcome['host_receipts_replayed'],0)

if __name__=='__main__':unittest.main()
