"""Source-pinned bounded STOP-margin probes; DEVELOPMENT only."""
from __future__ import annotations
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
import numpy as np
from . import dual_clock_lab as d
from . import stop_calibration_lab as s
from .world import MOVE, World, specs

class StopMarginTests(unittest.TestCase):
    def test_predeclared_controls_and_stop_is_still_offered(self):
        w=World.create(specs("train",1)[0]);m=d.Adapter(584)
        menu,p,choice,raw=s.score(w,"wave","dual_clock",m,2.5)
        self.assertIn("STOP",menu)
        self.assertEqual(len(menu),len(p))
        self.assertAlmostEqual(float(sum(p)),1.)
        self.assertIn(choice,menu)
        self.assertIn(raw,menu)
        self.assertGreater(float(p[menu.index("STOP")]),0.)
        with self.assertRaises(ValueError):
            s.score(w,"wave","dual_clock",m,77.)

    def test_margin_only_moves_stop_logit_when_a_move_is_legal(self):
        w=World.create(specs("train",1)[0]);m=d.Adapter(584)
        a,p0,_,_=s.score(w,"wave","dual_clock",m,0.)
        b,p1,_,_=s.score(w,"wave","dual_clock",m,2.5)
        self.assertEqual(a,b)
        self.assertTrue(any(k in MOVE for k in a))
        j=a.index("STOP")
        self.assertLess(float(p1[j]),float(p0[j]))
        non=[i for i,k in enumerate(a) if k!="STOP"]
        self.assertTrue(all(float(p1[i])>float(p0[i]) for i in non))

    def test_replay_probability_and_geometry_tampering(self):
        spec=specs("development",1)[0];m=d.Adapter(584)
        e=s.episode(spec,"wave_jag","dual_clock",1.0,m,128,True)
        self.assertGreater(s.verify_episode(e,m),0)
        f=deepcopy(e);f["decisions"][0]["p"][next(iter(f["decisions"][0]["p"]))]=.4321
        with self.assertRaises(ValueError):s.verify_episode(f,m)
        f=deepcopy(e);f["events"][0]["post_sha256"]="invalid"
        with self.assertRaises(ValueError):s.verify_episode(f,m)

    def test_no_train_or_heldout_evaluation_and_caps(self):
        m=d.Adapter(585)
        for split in ("train","heldout"):
            with self.assertRaises(ValueError):
                s.episode(specs(split,1)[0],"wave","dual_clock",0.,m,128,False)
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):
                s.run(Path(t)/"bad",seed=12345)
        with self.assertRaises(ValueError):
            s.variant_id("dual_clock",3.)

if __name__=="__main__":
    unittest.main()
