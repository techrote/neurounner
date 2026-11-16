"""Model-side, no native/Godot claims; run: python -m unittest discover -s . -p test_lab.py -v."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from lab import World, HostRefusal, TinyPolicy, train, episode, sha, evaluate, ARMS, INK, START, W


class HostContractTests(unittest.TestCase):
    def test_swept_prefix_stops_at_protected_intermediate(self):
        w=World(1,'empty','sweep')
        w.material[24*W+10]=5;w.protected.add((10,24))
        names,_=w.offer()
        self.assertIn('INK_E_1',names)
        self.assertNotIn('INK_E_2',names)
        self.assertNotIn('INK_E_3',names)
        self.assertIn('STOP',names)

    def test_stale_token_and_illegal_action_leave_material_untouched(self):
        w=World(2,'empty','sweep')
        names,token=w.offer()
        before=sha(w.material)
        with self.assertRaises(HostRefusal):w.apply('INK_E_2','bad-token')
        self.assertEqual(sha(w.material),before)
        with self.assertRaises(HostRefusal):w.apply('INK_E_5',token)
        self.assertEqual(sha(w.material),before)
        w.apply('INK_E_1',token)
        with self.assertRaises(HostRefusal):w.apply('INK_E_1',token)

    def test_lease_identity_cannot_cross_worlds(self):
        a=World(3,'empty','cell',lease='worm-A')
        b=World(3,'empty','cell',lease='worm-B')
        _,token=a.offer()
        with self.assertRaises(HostRefusal):b.apply('INK_E_1',token)
        self.assertEqual(b.decisions,0)

    def test_material_ownership_and_retrace_do_not_erase(self):
        w=World(4,'empty','retrace')
        _,token=w.offer();w.apply('INK_E_1',token)
        self.assertEqual(len(w.own),1)
        options,token=w.offer()
        self.assertIn('RETRACE_1',options)
        w.apply('RETRACE_1',token)
        self.assertEqual(w.cursor,START)
        self.assertEqual(len(w.own),1)
        self.assertEqual(w.material[24*W+9],INK)
        names,_=w.offer()
        self.assertIn('LIFT_E_1',names)
        self.assertNotIn('INK_E_1',names)

    def test_pen_up_refuses_foreign_material(self):
        w=World(5,'empty','lift')
        w.material[24*W+9]=3
        names,_=w.offer()
        self.assertNotIn('LIFT_E_1',names)
        self.assertNotIn('LIFT_E_2',names)
        self.assertNotIn('INK_E_1',names)

    def test_nested_ablation_legality(self):
        a={arm:set(World(6,'empty',arm).legal()) for arm in ARMS}
        for small,large in zip(ARMS,ARMS[1:]):
            self.assertLessEqual(a[small],a[large],(small,large))
        self.assertNotIn('SAVE_ANCHOR',a['retrace'])
        self.assertIn('SAVE_ANCHOR',a['anchor'])
        self.assertNotIn('JOG_N_E',a['anchor'])
        self.assertIn('JOG_N_E',a['jog'])

    def test_macro_validates_both_cells_not_only_destination(self):
        w=World(7,'empty','jog')
        w.material[23*W+8]=5;w.protected.add((8,23))
        names,_=w.offer()
        self.assertNotIn('JOG_N_E',names)
        self.assertIn('JOG_E_N',names)
        token=w.offer()[1]
        w.apply('JOG_E_N',token)
        self.assertEqual(len(w.own),2)
        self.assertEqual(w.logical_moves,2)

    def test_stop_is_terminal(self):
        w=World(8,'empty','jog')
        _,token=w.offer()
        w.apply('STOP',token)
        self.assertTrue(w.stopped)
        self.assertEqual(w.legal(),{})
        with self.assertRaises(HostRefusal):w.apply('INK_E_1',token)

    def test_budgets_enforced_without_native_limit_claim(self):
        w=World(9,'empty','sweep')
        w.logical_moves=384
        self.assertEqual(w.legal(),{'STOP':()})
        w=World(9,'empty','sweep')
        w.decisions=96
        self.assertEqual(w.legal(),{'STOP':()})
        w=World(9,'empty','sweep')
        w.own=set((i%W,i//W) for i in range(128))
        self.assertFalse(any(k.startswith('INK_') for k in w.legal()))

    def test_distinct_worlds_and_protection_stay_isolated(self):
        a=World(10,'pillars','sweep')
        b=World(10,'pillars','sweep')
        _,token=a.offer();a.apply('INK_E_2',token)
        self.assertNotEqual(a.material,b.material)
        self.assertEqual(a.original,b.original)
        self.assertEqual(b.own,set())
        self.assertEqual(a.protected,b.protected)

    def test_train_updates_and_heldout_traces_are_repeatable(self):
        p1,u1=train('cell',5);p2,u2=train('cell',5)
        self.assertEqual(p1.weights,p2.weights)
        self.assertEqual(u1,u2)
        self.assertGreater(u1,0)
        self.assertTrue(any(abs(x)>1e-8 for x in p1.weights))
        r1=evaluate(p1,'cell',count=5)
        r2=evaluate(p2,'cell',count=5)
        self.assertEqual(r1,r2)
        self.assertTrue(all(x['synthetic_only'] for x in r1))
        self.assertTrue(all(x['new_ink']<=128 and x['decisions']<=96 for x in r1))

if __name__=='__main__':unittest.main()
