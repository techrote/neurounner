"""Tests of paired guidance counterfactuals; no model/native execution."""
import json
from pathlib import Path
import tempfile
import unittest

from worm_sketch import canonical, digest, field, recipe, simulate
from worm_scent_study import ARMS, SCENES, SEEDS, corpus, guide_for, pair_summary, run_one, evidence_bytes


class ScentAblationTests(unittest.TestCase):
    def test_source_parity(self):
        for scene in SCENES:
            for seed in SEEDS:
                old = simulate(recipe(scene, seed, "wayward"))
                new = run_one(scene, seed, "attract")
                self.assertEqual(new["trail"], old["trail"])
                self.assertEqual(new["material"], old["material"])
                self.assertEqual(new["events"], old["events"])

    def test_fields_are_distinct_but_valid(self):
        for seed in SEEDS:
            bases = field(80, 80, seed)
            self.assertEqual(guide_for(80, 80, seed, "attract"), bases)
            self.assertEqual(guide_for(80, 80, seed, "attract_x16"), bases)
            for x, y in [(3, 9), (33, 42), (63, 61)]:
                self.assertEqual(guide_for(80, 80, seed, "repel")[x, y], 65535 - bases[x, y])
                self.assertEqual(guide_for(80, 80, seed, "mirror_x")[x, y], bases[79-x,y])
                self.assertEqual(guide_for(80, 80, seed, "neutral")[x,y], 32768)

    def test_hard_legality_and_mutation_receipts(self):
        for scene in SCENES:
            for arm in ARMS:
                out=run_one(scene,23,arm)
                ink=set()
                for e in out["events"]:
                    if e["kind"]!="STEP":
                        self.assertFalse(e.get("executed",False))
                        continue
                    p=tuple(e["to_cursor"])
                    old=tuple(e["from_cursor"])
                    self.assertEqual(sum(abs(p[i]-old[i]) for i in (0,1)),1)
                    self.assertIn(p,out["permitted"])
                    self.assertIn([e["heading"],list(p)],e["legal_catalogue"])
                    self.assertEqual(e["material_before_sha256"],digest(canonical(sorted(map(list,ink)))))
                    if e["pen"]=="down":ink.add(p)
                    self.assertEqual(e["material_after_sha256"],digest(canonical(sorted(map(list,ink)))))
                    self.assertIsNone(e["native_receipt"])
                    self.assertIsNone(e["model_decision"])
                self.assertEqual(ink,out["material"])

    def test_pairing_and_real_counterfactual_change(self):
        changed=0
        for scene in SCENES:
            for seed in SEEDS:
                neutral=run_one(scene,seed,"neutral")
                for arm in ARMS:
                    candidate=run_one(scene,seed,arm)
                    pair=pair_summary(neutral,candidate)
                    self.assertTrue(pair["same_hard_mask"])
                    self.assertTrue(pair["same_initial_cursor"])
                    if arm == "neutral":
                        self.assertIsNone(pair["first_divergence_step"])
                    elif pair["first_divergence_step"] is not None:
                        changed += 1
        self.assertGreaterEqual(changed,14)

    def test_deterministic_corpus_and_independent_recreation(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first=corpus(a)
            second=corpus(b)
            self.assertEqual(first,second)
            self.assertEqual(len(first["outcomes"]),36)
            self.assertEqual((Path(a)/"index.json").read_bytes(), (Path(b)/"index.json").read_bytes())
            for row in first["outcomes"]:
                folder=Path(a)/row["case"]
                scene, seed, arm=row["scene"],row["seed"],row["arm"]
                fresh=evidence_bytes(run_one(scene,seed,arm))
                manifest=json.loads((folder/"manifest.json").read_bytes())
                self.assertEqual(manifest["artifacts"],{n:digest(x) for n,x in fresh.items()})
                for n, data in fresh.items():
                    self.assertEqual((folder/n).read_bytes(),data)
                    self.assertEqual((Path(b)/row["case"]/n).read_bytes(),data)

    def test_admission_and_tamper_rejection(self):
        with self.assertRaises(ValueError):run_one("fake",23,"attract")
        with self.assertRaises(ValueError):run_one("ring",12,"attract")
        with self.assertRaises(ValueError):run_one("ring",23,"fake")
        with self.assertRaises(ValueError):run_one("ring",23,"attract",steps=513)
        with self.assertRaises(ValueError):run_one("ring",23,"attract",steps=0)
        with self.assertRaises(ValueError):run_one("ring",23,"attract",steps=True)
        with tempfile.TemporaryDirectory() as folder:
            corpus(folder)
            target=Path(folder)/"ring-seed23-mirror_x"/"after.png"
            original=target.read_bytes()
            target.write_bytes(original[:-1]+bytes([original[-1]^1]))
            regenerated=evidence_bytes(run_one("ring",23,"mirror_x"))
            self.assertNotEqual(target.read_bytes(),regenerated["after.png"])


if __name__=="__main__":unittest.main()
