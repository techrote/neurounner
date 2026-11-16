"""Small no-spend reproducibility and host-authority tests for C7 stochastic probes."""
import unittest
import torch
from experiment import Scorer
from positive_correction import CueGatedPolicy
from goal_switch_recovery import GoalSwitchAdapter
from personality_gate import BacktrackOnlyGate
from stochastic_probe import run

class StochasticProbeTests(unittest.TestCase):
    def models(self):
        torch.manual_seed(3)
        return {'C4':CueGatedPolicy(Scorer()).eval(),
                'C6':GoalSwitchAdapter(CueGatedPolicy(Scorer())).eval(),
                'C7':BacktrackOnlyGate(CueGatedPolicy(Scorer())).eval()}

    def test_probes_are_reproducible(self):
        models=self.models()
        a,sa=run(models,seeds=2,turns=5)
        b,sb=run(models,seeds=2,turns=5)
        self.assertEqual(a,b);self.assertEqual(sa,sb)
        self.assertEqual(len(a),3*2*3*2*3)
        self.assertEqual(sum(r['rejected'] for r in a),0)

    def test_source_identity_is_matched_per_arm(self):
        rows,_=run(self.models(),seeds=1,turns=3)
        src={(r['shape'],r['cue'],r['arm']):r['source_sha256'] for r in rows if r['model']=='C4'}
        self.assertTrue(all(r['source_sha256']==src[(r['shape'],r['cue'],r['arm'])] for r in rows))
        self.assertTrue(all(r['materials_sha256'] for r in rows))

if __name__=='__main__':unittest.main()
