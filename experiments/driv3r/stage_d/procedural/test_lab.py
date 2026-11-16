"""D-2 cheap deterministic tests: synthetic-only, no network, native or model."""
import json
from pathlib import Path
import tempfile
import unittest

from lab import (VERSION, Random32, hilbert, moore, minkowski, simulate,
                 default_recipes, write, replay)


class OfflineD2(unittest.TestCase):
    def test_prng_fixed_seed(self):
        a,b=Random32(42),Random32(42)
        self.assertEqual([a.next() for _ in range(24)],[b.next() for _ in range(24)])
        self.assertEqual(Random32(1).next(),270369)

    def test_full_curves_are_contiguous(self):
        for order in (2,3,4):
            for algorithm,side in ((hilbert,2**order),(moore,2**(order+1))):
                pts=algorithm(order)
                self.assertEqual(len(pts),side**2)
                self.assertEqual(len(set(pts)),len(pts))
                self.assertTrue(all(abs(a[0]-b[0])+abs(a[1]-b[1])==1
                                    for a,b in zip(pts,pts[1:])))
                if algorithm is moore:
                    self.assertEqual(abs(pts[0][0]-pts[-1][0])+
                                     abs(pts[0][1]-pts[-1][1]),1)

    def test_frozen_original_minkowski_silhouette(self):
        a=minkowski(32,32,32,1)
        b=minkowski(32,32,32,2)
        self.assertEqual(a[0],a[-1])
        self.assertEqual(b[0],b[-1])
        self.assertEqual(len(b)-1,8*(len(a)-1))
        self.assertNotEqual(a,b)

    def test_distinct_material_effects_and_sources(self):
        for recipe in default_recipes():
            r=simulate(recipe)
            self.assertGreater(r["summary"]["steps"],8)
            expected=-1 if recipe["family"]=="weathering_worm" else 1
            actual=r["summary"]["net_delta"]
            self.assertEqual((actual>0)-(actual<0),expected)
            self.assertTrue(r["geometry"])
            self.assertTrue(all(e.get("native_receipt") is None
                                for e in r["records"] if e["kind"]=="STEP"))
            self.assertTrue(any(e["kind"]=="UNDO_MATERIAL_EDIT_UNAVAILABLE" and
                                e["executed"] is False for e in r["records"]))

    def test_reversals_never_mutate_material(self):
        for recipe in default_recipes():
            for policy in ("backtrack","relocate"):
                r=simulate(dict(recipe,recovery=policy))
                for e in r["records"]:
                    if e["kind"]=="REVERSAL":
                        self.assertEqual(e["material_before_sha256"],
                                         e["material_after_sha256"])
                        self.assertIsNone(e["native_receipt"])
                        if e["semantics"]=="RELOCATE_TO_BRANCH":
                            self.assertEqual(e["reverse_trace"],[])
                        else:
                            t=e["reverse_trace"]
                            self.assertTrue(all(abs(a[0]-b[0])+abs(a[1]-b[1])<=1
                                                for a,b in zip(t,t[1:])))

    def test_hysteresis_reentry_and_no_retrace(self):
        r=simulate(default_recipes()[2])
        reentries=[e for e in r["records"] if e["kind"]=="GUIDANCE_REENTRY"]
        self.assertTrue(reentries)
        self.assertTrue(all(e["reentry_guidance_u16"]>=52000 and
                            e["lookahead_unvisited"]==4 for e in reentries))
        self.assertEqual(r["summary"]["visited"],r["summary"]["steps"])

    def test_contained_growth_does_not_overshoot(self):
        recipe=dict(default_recipes()[1],brush_radius=2,volume_radius=0)
        a=simulate(dict(recipe,containment="contained"))
        b=simulate(dict(recipe,containment="expressive"))
        def spill(x):
            return next(e["deposited_outside_authorized_volume"] for e in x["records"]
                        if e["kind"]=="GROWTH_END")
        self.assertEqual(spill(a),0)
        self.assertGreater(spill(b),0)

    def test_recovery_policies_have_identical_origins(self):
        for recipe in default_recipes():
            a=simulate(recipe)
            b=simulate(dict(recipe,recovery="relocate"))
            self.assertEqual(a["before"],b["before"])
            self.assertEqual(a["guidance"],b["guidance"])
            self.assertEqual(a["geometry"],b["geometry"])

    def test_original_files_and_replay(self):
        with tempfile.TemporaryDirectory() as root:
            for i,recipe in enumerate(default_recipes()):
                folder=Path(root)/str(i)
                manifest=write(recipe,folder)
                self.assertTrue(replay(folder))
                self.assertFalse(manifest["native_replay_verified"])
                self.assertEqual(manifest["qualification"],"OFFLINE_SYNTHETIC_ONLY")
                for file in ("before.png","midpoint.png","after.png","guidance.u16le",
                             "geometry.json","trajectory.json","events.json"):
                    self.assertIn(file,manifest["artifacts"])

    def test_tamper_refused(self):
        with tempfile.TemporaryDirectory() as root:
            write(default_recipes()[2],root)
            file=Path(root)/"events.json"
            file.write_bytes(file.read_bytes()+b" ")
            with self.assertRaises(AssertionError):
                replay(root)

    def test_invalid_budgets_and_unsupported_ops(self):
        example=default_recipes()[2]
        for invalid in (dict(example,steps=99999),dict(example,canvas=[999,999]),
                        dict(example,curve="not-a-curve"),
                        dict(example,exit_below=60000,reenter_at=55000)):
            with self.assertRaises((AssertionError,ValueError)):
                simulate(invalid)

    def test_reentry_has_explicit_unpainted_virtual_seek(self):
        for recovery in ("backtrack","relocate"):
            recipe=dict(default_recipes()[2],recovery=recovery)
            result=simulate(recipe)
            events=result["records"]
            seeks=[e for e in events if e["kind"]=="BRUSH_UP_SEEK_FUTURE"]
            enters=[e for e in events if e["kind"]=="GUIDANCE_REENTRY"]
            self.assertTrue(enters)
            self.assertEqual(len(seeks),len(enters))
            for seek,entry in zip(seeks,enters):
                self.assertEqual(seek["to"],
                                 result["geometry"]["ordered_curve"][entry["to_index"]])
                self.assertTrue(seek["virtual_offline_transfer"])
                self.assertFalse(seek["continuous_physical_path_verified"])
                self.assertTrue(seek["material_preserved"])
                self.assertEqual(seek["material_before_sha256"],
                                 seek["material_after_sha256"])
                self.assertIsNone(seek["native_receipt"])
            for event in events:
                if event["kind"]=="REVERSAL" and event["semantics"]=="BACKTRACK_CURSOR":
                    self.assertGreater(len(event["reverse_trace"]),1)

    def test_manifest_rehash_does_not_forge_replay(self):
        import hashlib
        from lab import packed
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)
            write(default_recipes()[2],path)
            file=path/"guidance-preview.png"
            changed=file.read_bytes()+b"NOT-A-REAL-PIXEL"
            file.write_bytes(changed)
            manifest=json.loads((path/"manifest.json").read_text())
            manifest["artifacts"]["guidance-preview.png"]={
                "bytes":len(changed),"sha256":hashlib.sha256(changed).hexdigest()}
            (path/"manifest.json").write_bytes(packed(manifest))
            with self.assertRaises(AssertionError):
                replay(path)

    def test_weathering_random_legal_control_is_source_matched(self):
        recipe=default_recipes()[0]
        a=simulate(dict(recipe,walker_policy="boundary_biased"))
        b=simulate(dict(recipe,walker_policy="random_legal"))
        self.assertEqual(a["before"],b["before"])
        self.assertEqual(a["guidance"],b["guidance"])
        self.assertEqual(a["geometry"],b["geometry"])
        self.assertTrue(any(e["kind"]=="SYNTHETIC_CHOICE" for e in a["records"]))
        for result in (a,b):
            for e in result["records"]:
                if e["kind"]=="SYNTHETIC_CHOICE":
                    self.assertIn(list(e["chosen"]),e["legal_points"])
                    self.assertFalse(e["native_legal_catalog"])
                    self.assertFalse(e["model_decision"])


if __name__=="__main__":
    unittest.main()
