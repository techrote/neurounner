"""Host-authority, observation boundary and reproducibility tests for GAL-009."""
import tempfile
import unittest
from pathlib import Path
from hard_observation import (ARMS, GoalWorld, Model, OBS_FEATURES, TRAIN_BASE,
    build_goal, observed, vector, teacher, train, eval_episode, run, audit)
from candidate_features import FAMILIES
from lab import START, HostRefusal, sha

class HardObservationTest(unittest.TestCase):
    def test_goals_are_host_clear_and_32(self):
        for kind in ('path','tree'):
            for family in FAMILIES:
                w=GoalWorld(TRAIN_BASE,family,'lift',kind,'test')
                self.assertEqual(32,len(w.goal))
                self.assertFalse(w.goal & w.protected)
                self.assertTrue(all(w.original[w.idx(p)]==0 for p in w.goal))

    def test_goals_are_seed_deterministic(self):
        for kind in ('path','tree'):
            self.assertEqual(build_goal(811234,'pillars',kind),build_goal(811234,'pillars',kind))

    def test_blind_observation_cannot_use_goal(self):
        w=GoalWorld(TRAIN_BASE,'empty','lift','tree','test')
        self.assertFalse(observed(w,'blind'))
        v={n:vector(w,n,p,'blind')[-len(OBS_FEATURES):] for n,p in w.legal().items()}
        for item in v.values():self.assertEqual(item[:7],(0.0,)*7)

    def test_local_observation_excludes_distant_target(self):
        w=GoalWorld(TRAIN_BASE,'empty','lift','path','test')
        w.goal=frozenset({(35,35)})
        self.assertFalse(observed(w,'local'))
        for n,p in w.legal().items():
            self.assertEqual(vector(w,n,p,'local')[:0],())
            self.assertEqual(vector(w,n,p,'local')[-8:-2],(0.0,)*6)
            self.assertEqual(vector(w,n,p,'local')[-2],0.0)

    def test_unseen_goal_not_leaked_to_local_vectors(self):
        w=GoalWorld(TRAIN_BASE,'empty','jog','tree','test')
        w.goal=frozenset({(35,35)})
        menu=w.legal()
        first={n:vector(w,n,p,'local') for n,p in menu.items()}
        w.goal=frozenset({(30,36),(42,40)})
        self.assertEqual(first,{n:vector(w,n,p,'local') for n,p in menu.items()})

    def test_same_legal_menu_across_sensor_views(self):
        w=GoalWorld(TRAIN_BASE,'gates','jog','tree','test')
        before=w.offer()
        for view in ('full','local','blind'):
            for name,path in before[0].items(): vector(w,name,path,view)
            self.assertEqual(before,w.offer())

    def test_host_refuses_stale_token_and_protected_writes(self):
        w=GoalWorld(TRAIN_BASE,'empty','jog','tree','test')
        menu,token=w.offer()
        with self.assertRaises(HostRefusal):w.apply('INK_E_1','stale')
        self.assertEqual(menu,w.legal())
        self.assertTrue(all(w.material[w.idx(p)]==w.original[w.idx(p)] for p in w.protected))

    def test_stop_offered_blind(self):
        w=GoalWorld(TRAIN_BASE,'empty','cell','path','test')
        self.assertIn('STOP',w.legal())
        self.assertEqual(len(vector(w,'STOP',(),'blind')),len(Model().weights))

    def test_learning_updates_and_replay(self):
        a,ra=train('cell','local',4);b,rb=train('cell','local',4)
        self.assertGreater(ra['updates'],0)
        self.assertEqual(a.weights,b.weights)
        self.assertTrue(any(abs(w)>1e-7 for w in a.weights))
        w,r=eval_episode(a,'cell','local','blind',920001,'tree')
        self.assertEqual(r['receipts_sha256'],sha(w.trace))

    def test_artifact_audit_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            run(tmp,train_worlds=4,heldout=4,gallery=0)
            self.assertEqual(audit(tmp)['episodes_replayed'],3*2*2*3*4)
            path=Path(tmp)/'episodes.json'
            path.write_text(path.read_text()+' ')
            with self.assertRaises(ValueError):audit(tmp)

if __name__=='__main__':unittest.main()
