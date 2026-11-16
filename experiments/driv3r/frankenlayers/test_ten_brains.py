"""M2 suite for 10 distinct source-ordered Frankenlayer controllers."""
import unittest
from experiments.driv3r.frankenlayers.patch_runtime import FLOWS


class TenBrains(unittest.TestCase):
    NAMES=("greyhound","branchwitch",
           "greyhound_restore5","greyhound_restore7",
           "greyhound_restore12","greyhound_restore16",
           "branchwitch_no_repeat10","branchwitch_restore8",
           "branchwitch_restore12","branchwitch_drop21")
    def test_ten_models_unique_and_scrambled_excluded(self):
        self.assertEqual(len(self.NAMES),10)
        assert len({FLOWS[k] for k in self.NAMES})==10
        self.assertNotIn("scrambled_eggs",self.NAMES)
        self.assertNotIn("stock",self.NAMES)
    def test_all_in_bounds_and_ending_real_final_block(self):
        for name in self.NAMES:
            with self.subTest(name=name):
                f=FLOWS[name]
                self.assertEqual(f[0],0)
                self.assertEqual(f[-1],23)
                self.assertTrue(all(0<=n<24 for n in f))
                inversions=[(a,b) for a,b in zip(f,f[1:]) if a>b]
                # BRANCHWITCH's original causal operation intentionally
                # executes layer 10 AGAIN after layer 11. Preserve that
                # source-qualified mutation rather than accidentally
                # imposing a sorting invariant that rejects its parent.
                self.assertEqual(inversions,[(11,10)] if
                                 name.startswith("branchwitch") and
                                 name!="branchwitch_no_repeat10" else [])
                self.assertGreaterEqual(len(f),13)
                self.assertLessEqual(len(f),21)
    def test_m2_mutants_are_one_change_from_original_flows(self):
        gh=FLOWS["greyhound"]
        bw=FLOWS["branchwitch"]
        for name in self.NAMES[2:6]:
            self.assertEqual(len(FLOWS[name]),len(gh)+1)
            self.assertEqual(tuple(n for n in FLOWS[name] if n in gh),gh)
        for name in self.NAMES[6:]:
            self.assertLessEqual(abs(len(FLOWS[name])-len(bw)),1)
