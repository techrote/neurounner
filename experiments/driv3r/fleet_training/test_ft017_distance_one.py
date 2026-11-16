"""FT-017 source/split/optimizer scientific control tests."""
import json
import unittest
from pathlib import Path
import numpy as np
import ft017_distance_one as ft

class TestFT017(unittest.TestCase):
    def test_request_and_hard_boundary(self):
        data=json.loads((Path(__file__).parent/"request_ft017.json").read_text())
        self.assertEqual(ft.check_request(data),ft.expected_request())
        self.assertEqual(data["native_material_edits"],0)
        with self.assertRaises(ValueError):
            ft.check_request({**data,"max_parallel_runner_jobs":6})
    def test_unique_leases(self):
        self.assertNotEqual(ft.lease(9401,"art","chaotic_brush",ft.TRAIN_ARMS[0]),
                            ft.lease(9401,"art","chaotic_brush",ft.TRAIN_ARMS[1]))
        with self.assertRaises(ValueError):
            ft.lease(9401,"navigation","chaotic_brush",ft.TRAIN_ARMS[0])
    def test_exactly_matched_training_examples(self):
        def x(z): return np.full(21,z,dtype=float)
        routes=[(1,[(x(2),0,3),(x(3),0,2),(x(5),0,1),(x(9),1,0)]),
                (2,[(x(4),0,4),(x(6),0,2),(x(7),0,1),(x(8),1,0)])]
        p,yp=ft.train_data(routes,ft.TRAIN_ARMS[0])
        n,yn=ft.train_data(routes,ft.TRAIN_ARMS[1])
        self.assertEqual(p.shape,(6,21))
        np.testing.assert_array_equal(yp,yn)
        np.testing.assert_array_equal(p[[0,1,3,4]],n[[0,1,3,4]])
        self.assertFalse(np.array_equal(p,n))
        self.assertEqual(sum(yp),2)
    def test_disallow_no_near_goal_negative(self):
        x=np.zeros(21)
        with self.assertRaises(AssertionError):
            ft.train_data([(1,[(x,0,3),(x,1,0)])],ft.TRAIN_ARMS[1])
    def test_goal_label_custody(self):
        x=np.zeros(21)
        a,b,d=ft.all_data([(1,[(x,0,1),(x,1,0)])])
        self.assertEqual(list(d),[1,0])
        with self.assertRaises(AssertionError):
            ft.all_data([(1,[(x,1,1),(x,1,0)])])
    def test_validation_calibration_false_positive_cap(self):
        class Gate:
            threshold=.5
            def probability(self,X): return X[:,0]
        X=np.array([[.9],[.8],[.4],[.2],[.1]])
        y=np.array([1,1,0,0,0])
        m=Gate()
        cal=ft.calibrate(m,X,y)
        self.assertEqual(cal["validation_recall"],1.0)
        self.assertEqual(cal["validation_fpr"],0.0)
        self.assertEqual(ft.heldout(m,X,y,np.array([0,0,1,2,3]))["distance_one_fp"],0)
    def test_budgets(self):
        self.assertEqual(ft.UPDATES,600)
        self.assertEqual(ft.MAX_RSS_KIB,900000)
        self.assertEqual(ft.AUTO,6)
