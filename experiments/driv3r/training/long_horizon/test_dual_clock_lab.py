"""Bounded dual-clock research tests; no sealed held-out worlds opened."""
from __future__ import annotations
from copy import deepcopy
import json
import tempfile
from pathlib import Path
import unittest
import numpy as np
from . import dual_clock_lab as d
from .world import World, specs

class DualClockTests(unittest.TestCase):
    def test_schedule_switch_exact(self):
        self.assertEqual(d.current_style('wave_jag',63),'wave')
        self.assertEqual(d.current_style('wave_jag',64),'jag')
        with self.assertRaises(ValueError):d.current_style('heldout',3)
    def test_new_memory_is_observed_only(self):
        w=World.create(specs('train',1)[0]);w.apply('MOVE_E')
        w.apply('MOVE_E');w.apply('BACKTRACK')
        self.assertEqual(len(d.headings(w,'executed')),2)
        self.assertEqual(len(d.headings(w,'live')),1)
        menu,a=d.features(w,'wave','executed_clock')
        menu2,b=d.features(w,'wave','dual_clock')
        self.assertEqual(menu,menu2)
        np.testing.assert_array_equal(a[:,28:],0)
        self.assertGreater(float(np.sum(np.abs(b[:,28:]))),0)
        self.assertEqual(w.recoveries,1)
    def test_train_data_exactly_paired(self):
        xs,ms,ys,pin=d.dataset(581,1,36)
        self.assertEqual(len(ys),len(ms))
        self.assertEqual(xs[d.ARMS[0]].shape,xs[d.ARMS[1]].shape)
        self.assertTrue(np.all(ms[np.arange(len(ys)),ys]))
        np.testing.assert_array_equal(xs['executed_clock'][:,:,:28],xs['dual_clock'][:,:,:28])
        self.assertEqual(pin,d.dataset(581,1,36)[-1])
    def test_real_optimizer_updates_and_checkpoint(self):
        xs,ms,ys,_=d.dataset(582,1,36)
        a=d.Adapter(582);b=d.Adapter(582)
        loss=a.train(xs['dual_clock'],ms,ys,3,900)
        self.assertEqual(loss['updates'],3)
        self.assertGreater(np.linalg.norm(a.w-b.w),0.)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'test.npz';a.save(p)
            r=d.Adapter.load(p)
            np.testing.assert_array_equal(r.w,a.w)
            self.assertEqual(r.updates,3)
    def test_forced_episode_replay_and_tampering(self):
        m=d.Adapter(581);spec=specs('development',1)[0]
        e=d.episode(spec,'wave_jag','dual_clock',m,128,True)
        self.assertGreater(d.verify_episode(e,m),0)
        f=deepcopy(e);f['scores'][0]['prob'][next(iter(f['scores'][0]['prob']))]=0.1234
        with self.assertRaises(ValueError):d.verify_episode(f,m)
        f=deepcopy(e);f['scores'][0]['active_style']='jag'
        with self.assertRaises(ValueError):d.verify_episode(f,m)
    def test_caps_and_no_unapproved_evaluation_split(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):d.run(Path(t)/'bad',worlds=50)
        with self.assertRaises(ValueError):d.episode(specs('train',1)[0],'wave','dual_clock',d.Adapter(1),128,False)

if __name__=='__main__':unittest.main()
