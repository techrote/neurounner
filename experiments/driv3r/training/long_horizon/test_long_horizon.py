"""Cheap source-only conformance, replay, split and real optimizer checks."""
import tempfile
import unittest
from pathlib import Path
import numpy as np

from .world import (World, Spec, MOVE, TRAIN_FAMILIES, HELDOUT_FAMILIES,
                    replay, specs)
from .policy import Ranker, teacher, menu_matrix, random_legal_persistent
from .train import make_corpus, episode, aggregate
import random

class WorldTests(unittest.TestCase):
    def test_structural_heldout_and_seed_seals(self):
        self.assertFalse(set(TRAIN_FAMILIES) & set(HELDOUT_FAMILIES))
        for split in ("train","development","heldout"):
            xs=specs(split,2)
            self.assertEqual(len(xs),len({s.ident for s in xs}))
            self.assertTrue(all(s.split==split for s in xs))
        with self.assertRaises(ValueError):
            Spec("train","rooms",1010,"motif")
        with self.assertRaises(ValueError):
            Spec("heldout","pickets",9000,"wall")
        with self.assertRaises(ValueError):
            Spec("train","ridges",9000,"route")

    def test_source_reproducible(self):
        s=specs("train",2)[0]
        a,b=World.create(s),World.create(s)
        self.assertEqual(a.base_sha(),b.base_sha())
        self.assertEqual(a.menu(),b.menu())
        self.assertIn("STOP",a.menu())
        self.assertEqual(a.state_sha(),b.state_sha())

    def test_one_cell_ink_legality_replay(self):
        w=World.create(specs("train")[0])
        for _ in range(30):
            candidate=next((a for a in w.menu() if a in MOVE),"STOP")
            if candidate == "STOP": break
            w.apply(candidate)
        x=replay(w.spec,w.decisions)
        self.assertEqual(x.ink,w.ink)
        self.assertEqual(len(w.ink),len([e for e in w.decisions if e["painted"]]))
        self.assertFalse(w.ink.intersection(w.materials))
        self.assertFalse(w.ink.intersection(w.protected))
        if w.decisions:
            corrupted = [dict(e) for e in w.decisions]
            corrupted[0]["painted"] = [[1,1]]
            with self.assertRaises(ValueError): replay(w.spec,corrupted)

    def test_recoveries_are_pen_up(self):
        w=World.create(specs("train")[0])
        for act in ["MOVE_E","MOVE_E","MOVE_E","MOVE_E","MOVE_E"]:
            if act in w.menu(): w.apply(act)
        if "BACKTRACK" in w.menu():
            count=len(w.ink)
            row=w.apply("BACKTRACK")
            self.assertEqual(row["painted"],[])
            self.assertEqual(len(w.ink),count)
            self.assertEqual(w.recoveries,1)
            self.assertEqual(replay(w.spec,w.decisions).state_sha(),w.state_sha())

    def test_cannot_ink_protected_or_twice(self):
        w=World.create(specs("train")[0])
        with self.assertRaises(ValueError):w.apply("INVALID")
        with self.assertRaises(ValueError):w.apply("BACKTRACK")
        w.ink.add((49,48))
        self.assertNotIn("MOVE_E",w.menu())
        with self.assertRaises(ValueError):w.apply("MOVE_E")
        w.protected.add((48,47))
        self.assertNotIn("MOVE_N",w.menu())
        with self.assertRaises(ValueError):w.apply("MOVE_N")

    def test_stop_never_masked(self):
        w=World.create(specs("train")[0])
        for _ in range(10):
            self.assertIn("STOP",w.menu())
            if "MOVE_E" in w.menu():w.apply("MOVE_E")
            else:break
        w.apply("STOP")
        self.assertTrue(w.stopped)
        self.assertEqual(w.menu(),[])

class TrainerTests(unittest.TestCase):
    def test_training_corpus_excludes_heldout_and_has_interventions(self):
        X,M,y,meta=make_corpus(36,570,2,adversarial_fraction=.45)
        self.assertEqual(len(y),meta["examples"])
        self.assertEqual(X.shape,(len(y),7,24))
        self.assertTrue(M[np.arange(len(y)),y].all())
        self.assertGreater(meta["off_policy_interventions"],0)
        self.assertTrue(all(row["family"] in TRAIN_FAMILIES for row in meta["worlds"]))

    def test_realtime_model_and_checkpoint(self):
        X,M,y,meta=make_corpus(36,570,1)
        p=Ranker(123)
        before=p.params["w1"].copy()
        stats=p.optimize(X,M,y,steps=35,batch=32,lr=.008,seed=42)
        self.assertGreater(stats["updates"],0)
        self.assertFalse(np.array_equal(before,p.params["w1"]))
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"ckpt.npz"
            p.checkpoint(path)
            q=Ranker.restore(path)
            self.assertTrue(np.array_equal(q.params["w1"],p.params["w1"]))
            r=episode(specs("development")[0],36,"trained_tiny_mlp",q,570)
            self.assertTrue(r["model_executed"])
            self.assertEqual(r["metrics"]["native_receipts"],0)
            self.assertTrue(all(e["chosen"] in e["legal"] for e in r["events"]))
            self.assertIn("STOP",r["events"][0]["legal"])

    def test_sequential_36_72_128_replay(self):
        s=specs("development")[0]
        for horizon in (36,72,128):
            e=episode(s,horizon,"teacher_mixture",None,570)
            self.assertLessEqual(e["metrics"]["decisions"],horizon)
            self.assertEqual(e["metrics"]["illegal_host_actions"],0)
            self.assertEqual(e["metrics"]["protected_cell_edits"],0)
            self.assertFalse(e["native_executed"])

    def test_random_stop_is_not_hidden(self):
        r=episode(specs("development")[0],128,"random_legal",None,570)
        self.assertTrue(all("STOP" in e["legal"] for e in r["events"]))
        self.assertLess(r["metrics"]["decisions"],128)

    def test_persistent_random_never_voluntarily_stops_during_progress(self):
        for spec in specs("development", 2):
            w = World.create(spec)
            rng = random.Random(71)
            for _ in range(72):
                menu = w.menu()
                action = random_legal_persistent(w, rng)
                self.assertIn(action, menu)
                self.assertNotEqual(action, "STOP") if any(a in MOVE for a in menu) else None
                w.apply(action)
                if w.stopped: break
            self.assertEqual(replay(spec, w.decisions).state_sha(), w.state_sha())

    def test_bad_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"invalid.npz"
            np.savez(path,seed=np.asarray(1),hidden=np.asarray(24),updates=np.asarray(1),
                     w1=np.asarray([float("nan")]),b1=np.zeros(24),w2=np.zeros(24),b2=np.zeros(1))
            with self.assertRaises(ValueError):Ranker.restore(path)

if __name__=="__main__": unittest.main()


class CurriculumAblationTests(unittest.TestCase):
    def test_equal_budget_horizon_only_ablation(self):
        from .compare_curricula import training_plan, train_policy
        self.assertEqual(training_plan("staged_36_72_128"), (36,72,128))
        self.assertEqual(training_plan("flat_128"), (128,128,128))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            staged,a = train_policy("staged_36_72_128", 573, 1, 3, p)
            flat,b = train_policy("flat_128", 573, 1, 3, p)
            self.assertEqual((a["updates"],b["updates"]), (9,9))
            self.assertEqual((a["parameters"],b["parameters"]), (625,625))
            self.assertNotEqual(a["checkpoint_sha256"],b["checkpoint_sha256"])
            self.assertFalse(np.array_equal(staged.params["w1"], flat.params["w1"]))

    def test_teacher_agreement_does_not_use_heldout(self):
        from .compare_curricula import teacher_conditional_agreement
        p = Ranker(2)
        dev = specs("development", 1)
        r = teacher_conditional_agreement(dev, p, 570)
        self.assertGreater(r["teacher_state_decisions"], 30)
        self.assertLessEqual(r["matches"], r["teacher_state_decisions"])


class ArtifactVerifierTests(unittest.TestCase):
    def test_dev_source_replay_verifier_rejects_missing_artifacts(self):
        from .verify_ablation import verify
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                verify(Path(d))

    def test_dev_ablation_seed_restriction(self):
        from .compare_curricula import training_plan
        with self.assertRaises(ValueError):
            training_plan("teacher_mixture")
