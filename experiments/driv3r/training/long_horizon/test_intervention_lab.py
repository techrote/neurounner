"""Independent limited adversarial tests for DEV-only host-controlled off-policy stress."""
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

from .intervention_lab import forced_choice, run_episode, train_arm, summarize
from .policy import Ranker
from .world import World, Spec, specs


class InterventionTests(unittest.TestCase):
    def test_offered_divergence_only(self):
        w=World.create(specs('development',1)[0])
        alternative=forced_choice(w,'MOVE_E',36)
        self.assertIn(alternative,w.menu())
        self.assertNotEqual(alternative,'MOVE_E')
    def test_observation_never_sees_heldout(self):
        s=Spec('heldout','archipelago',9000,'wall')
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                run_episode(s,None,'teacher_mixture',36,5,True,Path(td))
    def test_tampered_learner_credential_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                run_episode(specs('development',1)[0],None,'ordinary_offpolicy_013',36,7,True,Path(td))
    def test_real_optimizer_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            model,manifest=train_arm('stress_offpolicy_040',572,1,1,Path(td))
            self.assertEqual(model.updates,3)
            self.assertEqual(manifest['parameters'],625)
            self.assertEqual(len(manifest['sha256']),64)
    def test_original_trace_replays(self):
        with tempfile.TemporaryDirectory() as td:
            spec=replace(specs('development',1)[0],challenge='trap_corridor')
            a=run_episode(spec,None,'teacher_mixture',72,572,True,Path(td))
            self.assertEqual(a['native_receipts'],0)
            self.assertEqual(a['split'],'development')
            self.assertEqual(a['metrics']['protected_cell_edits'],0)
            self.assertEqual(a['metrics']['non_empty_cell_edits'],0)
            self.assertEqual(len(a['trajectory_sha256']),64)
            self.assertTrue((Path(td)/(f"{spec.ident}-72-teacher_mixture-1.json")).exists())
    def test_summary_negative_cases_not_suppressed(self):
        with tempfile.TemporaryDirectory() as td:
            spec=specs('development',1)[0]
            run=run_episode(spec,None,'random_legal_persistent',128,572,True,Path(td))
            s=summarize([run])
            self.assertEqual(sum(s['action_distribution'].values()),run['metrics']['decisions'])
            self.assertEqual(s['attempted_forces'],len(run['attempts']))
            self.assertEqual(s['actual_forces'],len(run['actual_interventions']))

if __name__=='__main__':unittest.main()

class VerificationAdversarialTests(unittest.TestCase):
    def test_forged_model_distribution_rejected(self):
        from .verify_intervention import verify_episode
        import copy
        with tempfile.TemporaryDirectory() as td:
            spec=specs('development',1)[0]
            model=Ranker(573)
            obj=run_episode(spec,model,'ordinary_offpolicy_013',36,573,True,Path(td))
            verified=verify_episode(obj,model)
            self.assertEqual(verified.state_sha(),obj['final_state_sha256'])
            forged=copy.deepcopy(obj)
            probs=forged['decision_probabilities'][0]['probabilities']
            keys=list(probs)
            probs[keys[0]]+=.25
            probs[keys[1]]-=.25
            with self.assertRaisesRegex(ValueError,'Forged decision probability'):
                verify_episode(forged,model)
    def test_rejected_fake_intervention_progress(self):
        from .verify_intervention import verify_episode
        import copy
        with tempfile.TemporaryDirectory() as td:
            spec=specs('development',1)[0]
            obj=run_episode(spec,None,'teacher_mixture',36,573,True,Path(td))
            forged=copy.deepcopy(obj)
            self.assertTrue(forged['actual_interventions'])
            forged['actual_interventions'][0]['new_ink_next_16_decisions']+=1
            with self.assertRaisesRegex(ValueError,'progress forged'):
                verify_episode(forged,None)
    def test_rejected_heldout_schema_forgery(self):
        from .verify_intervention import verify_episode
        with tempfile.TemporaryDirectory() as td:
            spec=specs('development',1)[0]
            obj=run_episode(spec,None,'teacher_mixture',36,573,True,Path(td))
            obj['split']='heldout'
            with self.assertRaisesRegex(ValueError,'Held-out'):
                verify_episode(obj,None)
