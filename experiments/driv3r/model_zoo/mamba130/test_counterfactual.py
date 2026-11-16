import unittest
from .counterfactual import directional_goals,target,samples,paired_switch_metrics,raw_features
from .world import ACTIONS,World

class CounterfactualTests(unittest.TestCase):
    def test_deterministic_and_seed_isolation(self):
        a=samples(range(300,305),10)
        self.assertEqual(a,samples(range(300,305),10))
        self.assertGreater(len(a),100)
        b=samples(range(900,904),10)
        self.assertFalse({r[2] for r in a}&{r[2] for r in b})

    def test_action_legality_under_goals(self):
        for seed in range(15):
            w=World.make(seed)
            for goal in directional_goals(w):
                self.assertTrue(w.safe(goal))
                w.goal=goal
                self.assertIn(target(w),w.legal())

    def test_goal_swaps_change_target_not_world(self):
        rows=samples(range(300,308),12)
        groups={}
        for prompt,lab,seed,t,k in rows:
            groups.setdefault(k,[]).append((prompt,lab))
        changed=0
        for rec in groups.values():
            prompts=[x[0] for x in rec]
            base=prompts[0].split('goal=')[0]+prompts[0].split(' t=')[1]
            for prompt in prompts:
                self.assertEqual(base,prompt.split('goal=')[0]+prompt.split(' t=')[1])
            changed+=int(len({x[1] for x in rec})>1)
        self.assertGreater(changed,20)

    def test_pair_metric_not_fooled_by_constant_class(self):
        rows=samples(range(300,303),5)
        oracle=paired_switch_metrics(rows,[x[1] for x in rows])
        constant=paired_switch_metrics(rows,[ACTIONS.index('BRANCH')]*len(rows))
        self.assertGreater(oracle['changed_target_pairs'],0)
        self.assertEqual(oracle['pair_accuracy'],1.)
        self.assertEqual(constant['pair_accuracy'],0.)
        self.assertEqual(constant['predicted_switch_rate'],0.)

    def test_structured_control_is_goal_sensitive(self):
        w=World.make(12);goals=directional_goals(w)
        w.goal=goals[0];a=raw_features(w.prompt())
        w.goal=goals[1];b=raw_features(w.prompt())
        self.assertEqual(len(a),114)
        self.assertEqual(len(b),114)
        self.assertNotEqual(a,b)
        self.assertEqual(a[:4],b[:4])
        self.assertEqual(a[7:],b[7:])

if __name__=='__main__':
    unittest.main()
