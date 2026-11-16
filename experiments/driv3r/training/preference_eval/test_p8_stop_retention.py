"""P8 tiny analytic STOP-coins: no fabricated W0 actions/artwork."""
import math
import unittest
from p8_stop_retention import expected_prefix,stop_probability,choice,archive,METRICS


def event(**kwargs):
    return {k:kwargs.get(k,0) for k in METRICS}


class P8STOPTests(unittest.TestCase):
    def test_no_stop_preserves_exact_source_prefix(self):
        a,e,r=expected_prefix([0,0],[event(new_positions=1,moves=1),event(branches=1,moves=1)])
        self.assertEqual(a,e)
        self.assertEqual(r,0)

    def test_stop_cannot_generate_future_art(self):
        a,e,r=expected_prefix([1,0],[event(new_positions=1,moves=1),event(new_positions=1,moves=1)])
        self.assertEqual(a['new_positions'],2)
        self.assertEqual(e['new_positions'],0)
        self.assertEqual(r,1)

    def test_closed_form_stop_probability(self):
        a,e,r=expected_prefix([.25,.5,0],[event(new_positions=1,moves=1),event(new_positions=1,moves=1),event(branches=1,moves=1)])
        self.assertAlmostEqual(r,.625)
        self.assertAlmostEqual(e['new_positions'],.75+.375)
        self.assertAlmostEqual(e['branches'],.375)

    def test_sample_and_greedy_are_distinct(self):
        logits=[0,0,0,0,0,2];legal=['STRAIGHT','STOP']
        self.assertEqual(stop_probability(logits,legal,0),1.)
        self.assertEqual(choice(logits,legal,0,0.8),'STOP')
        self.assertAlmostEqual(stop_probability([0]*6,legal,1),.5)

    def test_fail_closed_for_mutated_archive_or_nonfinite_logits(self):
        with self.assertRaises(ValueError):archive(b'not an authenticated archive')
        with self.assertRaises(ValueError):stop_probability([0,0,0,0,0,math.nan],['STRAIGHT','STOP'],1)
        with self.assertRaises(ValueError):stop_probability([0]*6,['STRAIGHT'],1)

    def test_fail_closed_for_invalid_probabilities_or_counts(self):
        for p in [math.nan,1.01,-.01,math.inf]:
            with self.assertRaises(ValueError):expected_prefix([p],[event(moves=1)])
        with self.assertRaises(ValueError):expected_prefix([.1],[])
        with self.assertRaises(ValueError):expected_prefix([.1],[event(moves=2)])

if __name__=='__main__':unittest.main()
