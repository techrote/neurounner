"""P12 source-fenced, model-side adversarial/receipt checks; no game imports."""
import importlib.util
import io
import json
import os
from pathlib import Path
import unittest
import zipfile

P=Path(__file__).with_name('p12_recovery_preference.py')
spec=importlib.util.spec_from_file_location('p12_recovery_preference',P)
p12=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p12)
ARCHIVE=Path(os.environ.get('P12_P11_ARTIFACT',str(Path(__file__).parent.parent/'immutable-P11-source'/'P11-hosted-43918309938.zip')))

class TestP12(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  if not ARCHIVE.exists():raise unittest.SkipTest('requires owner-pinned immutable P11 artifact')
  cls.bytes=ARCHIVE.read_bytes()
  cls.z,cls.w0,cls.head,cls.ck=p12.open_pinned(cls.bytes)

 def test_source_pin_rejects_mutation(self):
  corrupted=bytearray(self.bytes);corrupted[70]^=1
  with self.assertRaises(ValueError):p12.open_pinned(bytes(corrupted))

 def test_w16_model_host_contract(self):
  self.assertEqual(self.ck,'f5dc8482a145e39a0bfb10ba94617efad34dc957e3ff485afab3b13726fe90a9')
  r,srcsha,pngsha=p12.extract_trace(self.z,'architecture',2711)
  self.assertEqual(len(p12.model_features(r['events'][0]['observation'])),142)
  self.assertEqual(len(r['events']),len(r['head_decisions']))
  self.assertEqual(p12.sha(self.w0.png_bytes(r)),pngsha)

 def test_future_outcome_is_not_model_input(self):
  r,_,_=p12.extract_trace(self.z,'greebles',2711)
  n,b,z=p12.outcomes(r);s,c=p12.features(r,40,n,b,z)
  n2=list(n);n2[50:]=[1-v for v in n2[50:]]
  self.assertEqual(p12.features(r,40,n2,b,z),(s,c))
  self.assertNotEqual(sum(n[40:64]),sum(n2[40:64]))

 def test_seed_split_disjoint(self):
  self.assertFalse(set(p12.DISCOVERY)&set(p12.VALIDATION))
  self.assertFalse(set(p12.DISCOVERY)&set(p12.HELDOUT))
  self.assertFalse(set(p12.VALIDATION)&set(p12.HELDOUT))

 def test_tie_aware_precision(self):
  self.assertAlmostEqual(p12.average_precision([1,1],[1,0]),.5)
  self.assertAlmostEqual(p12.average_precision([0,0,0,0],[1,0,0,0]),.25)
  self.assertAlmostEqual(p12.average_precision([1,0],[1,0]),1.)

 def test_heldout_worlds_source_movement_exact(self):
  for f in p12.FAMILIES:
   r,_,_=p12.extract_trace(self.z,f,2711)
   live=p12.inference_rollout(self.w0,self.head,2711,f,r,'baseline')
   self.assertEqual(live['trajectory'],r['trajectory'])
   self.assertEqual(live['metrics'],r['metrics'])
   self.assertTrue(all(ev['raw_legal'] for ev in live['events']))

 def test_early_stop_is_always_source_prefix(self):
  r,_,_=p12.extract_trace(self.z,'greebles',2711)
  live=p12.inference_rollout(self.w0,self.head,2711,'greebles',r,'simple',threshold=1.)
  aa=[e['applied'] for e in r['events']]
  bb=[e['applied'] for e in live['events']]
  self.assertEqual(bb[-1],'STOP')
  self.assertEqual(bb[:-1],aa[:len(bb)-1])
  self.assertTrue(all(e['raw_legal'] for e in live['events']))

 def test_stop_score_does_not_impute_human_preference(self):
  r,_,_=p12.extract_trace(self.z,'greebles',2711)
  n=p12.score_stop(r,60)
  self.assertEqual(n['lost_novel']+n['novel'],r['metrics']['new_steps'])
  self.assertTrue(n['future_remaining_novel']>=n['future24_novel'])

if __name__=='__main__': unittest.main()
