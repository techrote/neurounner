"""CPU-only synthetic behavior and evidence-boundary tests."""
from __future__ import annotations
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .core import ACTIONS, DELTA, FAMILIES, FEATURES, World, teacher, choose_random, choose_wall, run, replay
from .train import generate, build_models, save_experiment


class HostContract(unittest.TestCase):
    def test_features_and_geometry_separate_from_family(self):
        worlds=[World(313,f) for f in FAMILIES]
        self.assertEqual(len(worlds[0].observe()),FEATURES)
        self.assertTrue(all(np.array_equal(worlds[0].blocked,w.blocked) for w in worlds))
        self.assertTrue(all(np.array_equal(worlds[0].protected,w.protected) for w in worlds))

    def test_illegal_and_protected_refused(self):
        w=World(1,'Pipework')
        with self.assertRaises(ValueError):w.apply(123)
        with self.assertRaises(ValueError):w.apply('UP')
        with self.assertRaises(ValueError):w.apply(True)
        # Explicitly test protected cardinal neighbor: host must reject even when
        # predicted by a neural policy. It never repairs an attempted bad action.
        w.protected[w.y,w.x+1]=True
        self.assertNotIn(3,w.legal())
        with self.assertRaises(ValueError):w.apply(3)
        self.assertEqual(int(w.visited.sum()),1)

    def test_dead_end_and_host_stop(self):
        w=World(4,'Architecture')
        for dx,dy in DELTA[:4]:w.blocked[w.y+dy,w.x+dx]=True
        self.assertEqual(w.legal(),(4,))
        self.assertEqual(teacher(w,__import__('random').Random(0)),4)
        w.apply(4)
        self.assertTrue(w.stopped)
        self.assertEqual(w.actions,['STOP'])
        self.assertEqual(int(w.visited.sum()),1)

    def test_exact_replay_and_tamper_refusal(self):
        for f in FAMILIES:
            w=World(337,f)
            evidence=run(w,choose_random,123)
            self.assertEqual(replay(evidence).state_sha256(),evidence['final_state_sha256'])
            tampered=copy.deepcopy(evidence)
            tampered['events'][1]['legal']=['STOP']
            with self.assertRaises(ValueError):replay(tampered)
            tampered=copy.deepcopy(evidence)
            tampered['events'][0]['state_sha256']='0'*64
            with self.assertRaises(ValueError):replay(tampered)

    def test_seeded_teacher_and_wall_replay(self):
        for selector in [teacher,choose_wall]:
            a=run(World(359,'Greebles'),selector,527)
            b=run(World(359,'Greebles'),selector,527)
            self.assertEqual(a,b)
            self.assertEqual(len(a['events']),96)

    def test_world_group_isolation(self):
        x,y,m,ids=generate([1001,1002])
        self.assertEqual(len(ids),len(x))
        self.assertEqual(len(x),2*3*96)
        self.assertEqual(set(s for s,_,_,_,_ in ids),{1001,1002})
        self.assertTrue(m[np.arange(len(m)),y].all())
        self.assertTrue(np.isfinite(x).all())

    def test_split_leakage_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'no-write'
            with self.assertRaises(ValueError):
                save_experiment(p,train_seeds=[3],dev_seeds=[3],test_seeds=[4],epochs=1)
            self.assertFalse(p.exists())

    def test_compact_models_load_and_masked_choices(self):
        import torch
        x=torch.randn(4,FEATURES)
        models=build_models(torch)
        self.assertEqual(set(models),{'linear','mlp','conv'})
        for model in models.values():
            with torch.no_grad(): logits=model(x)
            self.assertEqual(tuple(logits.shape),(4,5))
            self.assertTrue(torch.isfinite(logits).all())
            mask=torch.tensor([[False,False,True,False,True]]*4)
            pred=logits.masked_fill(~mask,-1e5).argmax(dim=1)
            self.assertTrue(all(i in (2,4) for i in pred.tolist()))


if __name__=='__main__':unittest.main()
