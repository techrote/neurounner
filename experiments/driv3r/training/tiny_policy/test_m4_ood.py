"""M4 finite positive/adversarial tests on strictly synthetic action receipts."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from .core import FAMILIES
from .m4_ood import SEEDS, TOPOLOGIES, geometry_hash, load_heads, novel, reachable, replay, rollout, world


class TestM4OOD(unittest.TestCase):
    def test_disjoint_and_geometry_family_invariant(self):
        assert len(SEEDS)==len(set(SEEDS))==12
        assert set(SEEDS).isdisjoint(range(7300,7340))
        assert set(SEEDS).isdisjoint(range(8300,8308))
        assert set(SEEDS).isdisjoint(range(9300,9308))
        for seed in SEEDS:
            ws=[world(seed,family) for family in FAMILIES]
            self.assertEqual(len(set(geometry_hash(w) for w in ws)),1)
            self.assertGreaterEqual(reachable(ws[0]),30)
            self.assertTrue(any(a!=4 for a in ws[0].legal()))

    def test_positive_replay_and_repeated_seed(self):
        a,w1=rollout(SEEDS[0],FAMILIES[0],novel,'novel-greedy')
        b,w2=rollout(SEEDS[0],FAMILIES[0],novel,'novel-greedy')
        self.assertEqual(a,b)
        self.assertEqual(w1.visited.tobytes(),w2.visited.tobytes())
        self.assertEqual(w1.visited.tobytes(),replay(a).visited.tobytes())

    def test_illegal_and_tamper_rejected(self):
        a,_=rollout(SEEDS[2],FAMILIES[1],novel,'novel-greedy')
        for key,modified in [('legal', ['STOP']),('action','BOGUS'),('feature_sha256','0'*64),('before_sha256','0'*64)]:
            bad=copy.deepcopy(a);bad['events'][0][key]=modified
            with self.assertRaises((ValueError,KeyError)):replay(bad)
        bad=copy.deepcopy(a);bad['geometry_sha256']='0'*64
        with self.assertRaises(ValueError):replay(bad)
        bad=copy.deepcopy(a);bad['cells']+=1
        with self.assertRaises(ValueError):replay(bad)

    def test_immutable_out(self):
        from .m4_ood import experiment
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):experiment(Path(d),Path(d))

    def test_checkpoint_hash_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            for name in ('linear','mlp','conv'):(path/(name+'.pt')).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):load_heads(path)

if __name__=='__main__':unittest.main()
