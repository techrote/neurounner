"""P9 adversarial checks: no future labels, no human votes, no game IO."""
import math
import unittest

from p9_clock_structure import (clock_features,local_features,vector,fit,predict,
                                cross_entropy,sigmoid,source_episodes,MODES,evaluate)


class ClockStructureP9Tests(unittest.TestCase):
    def test_clock_is_not_a_handpicked_pulse_flag(self):
        self.assertEqual(len(clock_features(88)),34)
        self.assertEqual(clock_features(88)[:2],[88/127,(88/127)**2])
        with self.assertRaises(ValueError):clock_features(-1)
        with self.assertRaises(ValueError):clock_features(128)

    def test_spatial_features_only_observe_prior_moves(self):
        obs={'x':18,'y':23,'heading':2,'legal':['STRAIGHT','LEFT','STOP'],
             'local_5x5':['.....','..#..','..@+.','.....','.....']}
        x=local_features(obs,['STRAIGHT','BRANCH'],{(0,0),(1,0)},[1,0],1)
        self.assertEqual(len(x),15)
        self.assertAlmostEqual(x[3],.5)
        self.assertAlmostEqual(x[6],.5)
        self.assertEqual(x[-4:],[0,0,1,0])
        self.assertEqual(local_features(obs,['STRAIGHT','BRANCH'],{(0,0),(1,0)},[1,0],1),x)
        for corrupt in (['..@..','..@..','..@..','..@..','..@..'], ['..X..']*5):
            with self.assertRaises(ValueError):local_features({**obs,'local_5x5':corrupt},[],{(0,0)},[],0)

    def test_model_features_are_explicit(self):
        row={'mode':MODES[0],'clock':clock_features(32),'space':[0.1]*15}
        self.assertEqual(len(vector(row,'mode')),3)
        self.assertEqual(len(vector(row,'clock')),37)
        self.assertEqual(len(vector(row,'space')),18)
        self.assertEqual(len(vector(row,'combined')),52)
        with self.assertRaises(ValueError):vector(row,'unknown')

    def test_real_adam_updates_improve_simple_soft_target(self):
        rows=[{'mode':MODES[i%3], 'clock':clock_features(i%128),
               'space':[i%7/7]*15,'stop_p':.6 if i%3==0 else .02} for i in range(90)]
        head=fit(rows,'mode',epochs=70)
        initial=sum(cross_entropy(r['stop_p'],sum(x['stop_p'] for x in rows)/len(rows)) for r in rows)/len(rows)
        final=sum(cross_entropy(r['stop_p'],predict(head,r)) for r in rows)/len(rows)
        self.assertLess(final,initial)
        self.assertTrue(any(abs(v)>1e-6 for v in head['weights']))
        self.assertEqual(head['updates'],70)

    def test_fail_closed_for_wrong_archive_and_bad_labels(self):
        with self.assertRaises(ValueError):source_episodes(b'unpinned source')
        with self.assertRaises(ValueError):fit([], 'mode')
        with self.assertRaises(ValueError):fit([{'mode':MODES[0], 'stop_p':math.nan}], 'mode')
        with self.assertRaises(ValueError):evaluate([], 'tick')

    def test_true_soft_brier_and_cross_entropy_have_stable_edges(self):
        self.assertTrue(math.isfinite(cross_entropy(.2,0)))
        self.assertTrue(math.isfinite(cross_entropy(.8,1)))
        self.assertAlmostEqual(sigmoid(0),.5)
        self.assertLess(sigmoid(-800),1e-100)
        self.assertGreater(sigmoid(800),.9999)


if __name__=='__main__':unittest.main()
