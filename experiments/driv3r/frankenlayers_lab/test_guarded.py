"""Wave 17 causal shield, geometry isolation and fail-closed legality tests."""
import unittest
from experiments.driv3r.frankenlayers_lab import counterfactual as cf
from experiments.driv3r.frankenlayers_lab.infer_guarded import (
    EVAL_SEED,EVAL_FAMILY,evaluation_obstacles,heldout,guarded_predict)
from experiments.driv3r.frankenlayers_lab.repair_adapter import N_FEATURES,ACTIONS

class GuardedTests(unittest.TestCase):
    def test_guard_preserves_raw_at_obstacles_but_repairs_open_and_stop(self):
        weights=[0.]*N_FEATURES
        weights[ACTIONS.index("LEFT")*9]=20.
        ckpt={"schema":"frankenlayers-frozen-d1-residual/1","weights":weights}
        obs={"family":"greebles","x":7,"y":48,"heading":0,"tick":0,
             "local_5x5":[".....",".....","..@..",".....","....."],
             "last_actions":[],"legal":["STRAIGHT","LEFT","RIGHT","STOP"]}
        p={"STRAIGHT":.15,"LEFT":.15,"RIGHT":.6,"STOP":.1}
        a,d=guarded_predict(obs,"RIGHT",p,ckpt)
        self.assertEqual(a,"LEFT")
        self.assertTrue(d["overrode"])
        obs["local_5x5"][0]="###.."
        a,d=guarded_predict(obs,"RIGHT",p,ckpt)
        self.assertEqual(a,"RIGHT")
        self.assertTrue(d["shielded"])
        a,d=guarded_predict(obs,"STOP",p,ckpt)
        self.assertEqual(a,"LEFT")
        self.assertFalse(d["shielded"])
        self.assertIn(a,obs["legal"])

    def test_heldout_restores_original_host(self):
        old=(cf.SEED,cf.FAMILY,cf.variant_obstacles)
        with heldout():
            self.assertEqual((cf.SEED,cf.FAMILY),(EVAL_SEED,EVAL_FAMILY))
            self.assertEqual(cf.variant_obstacles("east_blocked"),{(10,48)})
            self.assertEqual(cf.variant_obstacles("north_blocked"),{(8,47),(9,47)})
            self.assertNotEqual(cf.variant_obstacles("north_blocked"),cf.variant_obstacles("south_blocked"))
        self.assertEqual((cf.SEED,cf.FAMILY,cf.variant_obstacles),old)
        with self.assertRaises(ValueError):
            evaluation_obstacles("fabricated")

if __name__=="__main__":unittest.main()
