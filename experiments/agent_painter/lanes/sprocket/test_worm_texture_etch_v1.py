#!/usr/bin/env python3
"""Source-only tests for derived texture art, never fake training evidence."""
from __future__ import annotations
import copy
import importlib.util
from pathlib import Path
import struct
import unittest

SPEC=importlib.util.spec_from_file_location('worm_texture_etch_v1',Path(__file__).with_name('worm_texture_etch_v1.py'))
etch=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(etch)

class TextureReplayTest(unittest.TestCase):
    def fixture(self,family='greebles'):
        path=[[2,2],[2,3],[3,3],[3,2]]
        return {'contract':'cybersand.worm.observation-action/0-experimental',
          'size':96,'seed':11,'family':family,'mode':'learned-original-W0',
          'trajectory':path,'obstacles':[[0,0],[50,50]],'steps_executed':3,
          'events':[{'observation':{'x':path[i][0],'y':path[i][1],
                            'legal':['STRAIGHT','LEFT','BRANCH','RETRACE','STOP']},
                     'applied':action,'position_after':path[i+1],
                     'raw_choice':action,'raw_legal':True}
                    for i,action in enumerate(['BRANCH','LEFT','RETRACE'])],
          'metrics':{'invalid_raw_choices':0}}
    def test_deterministic_byte_identical(self):
        for family in etch.INK:
            obj=self.fixture(family)
            initial=copy.deepcopy(obj)
            a,ma=etch.draw(obj);b,mb=etch.draw(obj)
            self.assertEqual(a,b)
            self.assertEqual(ma,mb)
            self.assertEqual(obj,initial)
            self.assertTrue(a.startswith(b'\x89PNG\r\n\x1a\n'))
            w,h=struct.unpack('!2I',a[16:24])
            self.assertEqual((w,h),(768,768))
            self.assertTrue(ma['derived_not_original_W0'])
            self.assertTrue(ma['no_model_inference_in_replay'])
    def test_reject_tampered_path(self):
        obj=self.fixture();obj['trajectory'][1]=[2,5]
        with self.assertRaises(AssertionError): etch.draw(obj)
    def test_reject_obstacle_occupancy(self):
        obj=self.fixture();obj['obstacles'].append([2,3])
        with self.assertRaises(AssertionError): etch.draw(obj)
    def test_reject_illegal_choice(self):
        obj=self.fixture();obj['events'][0]['applied']='TELEPORT'
        with self.assertRaises(AssertionError): etch.draw(obj)
    def test_reject_false_event_lineage(self):
        obj=self.fixture();obj['events'][0]['position_after']=[8,8]
        with self.assertRaises(AssertionError): etch.draw(obj)

if __name__=='__main__':unittest.main()
