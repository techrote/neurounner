"""Cheap, optimiser-free M2 geometry, legality and evidence replay qualification."""
import copy
import unittest
import numpy as np
from .core import FAMILIES, teacher, choose_random, choose_wall
from .m2_transfer import CASES, SPLITS, WINDOW, world, trace, replay

class M2Host(unittest.TestCase):
    def test_split_isolation_and_geometry(self):
        self.assertFalse({s for s,_ in CASES} & set(sum(SPLITS.values(), [])))
        for seed,case in CASES:
            w=[world(seed,f,case) for f in FAMILIES]
            for x in w:
                self.assertEqual(x.budget,96)
                self.assertEqual(len(x.observe()),162)
                self.assertIn(4,x.legal())
                self.assertFalse(x.blocked[48,48] or x.protected[48,48])
            self.assertTrue(all(np.array_equal(w[0].blocked,x.blocked) and
                                np.array_equal(w[0].protected,x.protected) for x in w[1:]))

    def test_control_replay_all_cases_and_families(self):
        for seed,case in CASES:
            for family in FAMILIES:
                for name,fn in (("teacher",teacher),("random",choose_random),("wall",choose_wall)):
                    original=trace(world(seed,family,case),case,name,fn)
                    self.assertLessEqual(len(original["events"]),WINDOW)
                    self.assertTrue(all(e["action"] in e["legal"] for e in original["events"]))
                    self.assertEqual(original,trace(world(seed,family,case),case,name,fn))
                    self.assertEqual(replay(original).state_sha256(),original["final_sha256"])

    def test_fail_closed_corruption(self):
        seed,case=CASES[0]
        original=trace(world(seed,"Pipework",case),case,"random",choose_random)
        for field,replacement in (("geometry_sha256","0"*64),
                                  ("stop",not original["stop"]),
                                  ("final_sha256","0"*64)):
            bad=copy.deepcopy(original)
            bad[field]=replacement
            with self.assertRaises(ValueError): replay(bad)
        bad=copy.deepcopy(original)
        bad["events"][0]["legal"]=["STOP"]
        with self.assertRaises(ValueError): replay(bad)

    def test_protected_neighbor_not_host_legal(self):
        w=world(6163,"Architecture","protected_gate")
        w.protected[w.y,w.x+1]=True
        self.assertNotIn(3,w.legal())
        with self.assertRaises(ValueError):w.apply(3)

if __name__=="__main__":unittest.main()
