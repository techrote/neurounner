import copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
import ncr_001_void_mechanical_complete_adjudication as gate
class T(unittest.TestCase):
 def test_positive(self):
  r=gate.build();self.assertEqual(r["status"],gate.STATUS);self.assertTrue(r["masked_B2_incremental_cache_measured"]);self.assertIsNone(r["known_locked_72_mechanical_evidence_gap"]);self.assertFalse(r["per_model_conformance_pass"]);self.assertFalse(r["train_authorized"])
 def test_prior_final_admission_stays_false(self):
  orig=gate.prior.build_report
  x=orig();x["per_model_conformance_pass"]=True
  with unittest.mock.patch.object(gate.prior,"build_report",return_value=x):
   with self.assertRaisesRegex(ValueError,"REFUSED"):gate.build()
 def test_masked_values_are_bounded(self):
  r=gate.build();self.assertLessEqual(r["masked_B2_left_full_vs_cached_max_abs"],5e-4);self.assertLessEqual(r["masked_B2_right_full_vs_cached_max_abs"],5e-4)
 def test_remaining_items_are_not_retroactive_mechanical_checks(self):
  r=gate.build();self.assertIn("fresh_independent_host_replication_not_measured",r["remaining_items_not_retroactive_new_72_mechanical_checks"]);self.assertIn("explicit_final_model_specific_scientific_72_admission_not_granted",r["remaining_scientific_or_downstream_gaps"])
 def test_committed_decision_when_present(self):
  if not gate.OUTPUT.exists(): self.skipTest("first pass")
  self.assertEqual(json.loads(gate.OUTPUT.read_text()),gate.build())
if __name__=="__main__":unittest.main()
