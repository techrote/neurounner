"""Cheap leakage, legality, objective and Adam/checkpoint tests for history adapter."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from .history_motif_lab import Adapter, STYLES, history, matrix, teacher, corpus, episode, main
from .world import World, specs, replay

class HistoryMotifTests(unittest.TestCase):
    def test_host_observation_immutable(self):
        w=World.create(specs('development',1)[0]); pre=w.state_sha()
        for style in STYLES:
            for blind in (True,False):
                menu,X=matrix(w,style,blind)
                self.assertEqual(menu,w.menu())
                self.assertTrue(np.isfinite(X).all())
                self.assertEqual(w.state_sha(),pre)
    def test_blind_exactly_blinds_history(self):
        w=World.create(specs('development',1)[0]);menu,X=matrix(w,'arc',False)
        menu2,Y=matrix(w,'arc',True)
        self.assertEqual(menu,menu2)
        np.testing.assert_array_equal(X[:,:15],Y[:,:15])
        np.testing.assert_equal(Y[:,15:],0)
    def test_training_is_partitioned(self):
        X,M,Y,sig=corpus(575,1,36)
        self.assertEqual(X.shape[1:],(7,19))
        self.assertEqual(len(Y),len(M))
        self.assertTrue(np.all(M[np.arange(len(Y)),Y]))
        self.assertEqual(len(sig),64)
    def test_heldout_forbidden(self):
        m=Adapter(575,False)
        with self.assertRaises(ValueError):episode(specs('heldout',1)[0],'arc',m,36)
    def test_checkpoint_restorable(self):
        m=Adapter(575,False)
        X,M,Y,_=corpus(575,1,36)
        before=m.weights.copy();m.optimize(X,M,Y,2,575)
        self.assertFalse(np.array_equal(before,m.weights))
        with tempfile.TemporaryDirectory() as t:
            f=Path(t)/'w.npz';m.save(f);other=Adapter.load(f)
            np.testing.assert_array_equal(other.weights,m.weights)
            self.assertEqual(other.updates,2)
    def test_actual_source_replay_and_probabilities(self):
        s=specs('development',1)[0];m=Adapter(575,False)
        r=episode(s,'arc',m,36,force=True)
        self.assertEqual(r['native_receipts'],0)
        self.assertEqual(r['final_sha256'],replay(s,r['events']).state_sha())
        self.assertTrue(all(set(p['distribution'])==set(e['legal'])
                            for p,e in zip(r['probabilities'],r['events'])))
    def test_immutable_campaign_smoke(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)/'run';r=main(['--output',str(run),'--updates','1','--worlds','1'])
            self.assertEqual(r['episodes'],112)
            self.assertEqual(set(r['checkpoints']),{'history_aware','history_blind'})
            self.assertFalse(r['native_executed'])
            with self.assertRaises(FileExistsError):main(['--output',str(run),'--updates','1'])
if __name__=='__main__':unittest.main()
