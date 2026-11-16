"""Source-only tests for actual frozen-base residual training and isolation."""
import math
import unittest
from experiments.driv3r.frankenlayers_lab import counterfactual as cf
from experiments.driv3r.frankenlayers_lab.infer_repair import (
    TRAIN_SEED,EVAL_SEED,TRAIN_FAMILY,EVAL_FAMILY,heldout,evaluation_obstacles)
from experiments.driv3r.frankenlayers_lab.repair_adapter import (
    N_FEATURES,EPOCHS,train,predict,target,probs)

def example(proposal="STOP",legal=None,rows=None):
    legal=legal or ["STRAIGHT","LEFT","RIGHT","STOP"]
    obs={"family":"architecture","heading":0,"x":7,"y":48,"tick":7,
         "local_5x5":rows or [".....",".....","..@..",".....","....."],
         "last_actions":["LEFT","RIGHT"],"legal":legal}
    n=len(legal)
    p={a:(.52 if a==proposal else (.48/(n-1) if n>1 else 0)) for a in legal}
    return {"observation":obs,"choice":proposal,"probabilities":p}

class AdapterTests(unittest.TestCase):
    def test_training_determinism_finite_weights_and_legality(self):
        data=[example(),example("RIGHT"),example("LEFT"),
              example("STOP",rows=[".....",".....","..@##",".....","....."]),
              example("STOP",legal=["STOP"])]
        a=train(data);b=train(data)
        self.assertEqual(a,b)
        self.assertEqual((a["source_records"],a["epochs"]), (5,EPOCHS))
        self.assertEqual(len(a["weights"]),N_FEATURES)
        self.assertTrue(all(math.isfinite(v) for v in a["weights"]))
        self.assertTrue(a["base_systemone_frozen"])
        for case in data:
            action,detail=predict(case["observation"],case["choice"],case["probabilities"],a)
            self.assertIn(action,case["observation"]["legal"])
            self.assertEqual(detail["overrode"],action!=case["choice"])

    def test_supervised_retrace_not_premature_stop(self):
        case=example()
        self.assertNotEqual(target(case["observation"],case["choice"],case["probabilities"]),"STOP")
        case=example("STOP",legal=["RETRACE","STOP"],
                     rows=["#####","#####","##@##","#####","#####"])
        self.assertEqual(target(case["observation"],"STOP",case["probabilities"]),"RETRACE")
        case=example("STOP",legal=["STOP"])
        self.assertEqual(target(case["observation"],"STOP",case["probabilities"]),"STOP")

    def test_fail_closed_on_wrong_model_probability_support(self):
        c=example();a=train([c])
        with self.assertRaises(ValueError):
            predict(c["observation"],c["choice"],{"STOP":1.},a)
        with self.assertRaises(ValueError):
            probs({"STOP":float("nan")},["STOP"])

    def test_seed_family_geometry_isolation_and_restoration(self):
        self.assertEqual((TRAIN_SEED,EVAL_SEED),(2718,3891))
        self.assertEqual((TRAIN_FAMILY,EVAL_FAMILY),("architecture","pipework"))
        old=(cf.SEED,cf.FAMILY,cf.variant_obstacles)
        with heldout():
            self.assertEqual((cf.SEED,cf.FAMILY),(3891,"pipework"))
            self.assertEqual(cf.variant_obstacles("east_blocked"),{(10,48)})
            self.assertEqual(evaluation_obstacles("south_blocked"),{(8,50),(9,50)})
        self.assertEqual((cf.SEED,cf.FAMILY,cf.variant_obstacles),old)
if __name__=="__main__":unittest.main()
