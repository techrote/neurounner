"""Cheap route/custody/adverse tests for D-2 synthetic recovery (no native/model)."""
import json
from pathlib import Path
import tempfile
import unittest

from recovery_study import (read_episode, shortest_route, run_arm, study,
                            components, encoded, digest)


ROOT = Path(__file__).parent / "qualified-corpus"


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._scratch = None
        if ROOT.exists():
            cls.source = ROOT
        else:
            # An exact-source checkout creates its own ephemeral DEVELOPMENT
            # fixtures. Nothing is added to Git or sent to Actions.
            from runner import run
            cls._scratch = tempfile.TemporaryDirectory()
            cls.source = Path(cls._scratch.name) / "qualified-corpus"
            catalogue = Path(__file__).parent / "corpus/development-recipes.json"
            run(catalogue, cls.source)

    @classmethod
    def tearDownClass(cls):
        if cls._scratch is not None:
            cls._scratch.cleanup()

    def test_route_cardinal_and_deterministic(self):
        allowed = {(x,y) for x in range(4) for y in range(4)} - {(1,0),(1,1),(1,2)}
        expected = [(0,0),(0,1),(0,2),(0,3),(1,3),(2,3),(2,2),(2,1),(2,0)]
        self.assertEqual(shortest_route((0,0),(2,0),allowed,8),expected)
        self.assertIsNone(shortest_route((0,0),(2,0),allowed,7))
        self.assertEqual(shortest_route((0,0),(0,0),allowed,0),[(0,0)])

    def test_blocked_and_missing_endpoints(self):
        allowed = {(0,0),(0,1),(2,0),(2,1)}
        self.assertIsNone(shortest_route((0,0),(2,0),allowed,20))
        self.assertIsNone(shortest_route((0,0),(9,9),allowed,20))
        self.assertIsNone(shortest_route((9,9),(0,0),allowed,20))

    def test_components(self):
        self.assertEqual(components([]),[])
        self.assertEqual(components({(0,0),(1,0),(7,7)}),[2,1])

    def test_original_replay_and_identical_initial_fixture(self):
        for name in ("hilbert-reentry","moore-reentry","serpentine-reentry"):
            folder = self.source / name
            recipe, curve, scalar, _ = read_episode(folder)
            original = json.loads((folder/"manifest.json").read_bytes())
            virtual = run_arm(recipe,curve,scalar,"virtual_future")
            routed = run_arm(recipe,curve,scalar,"routed_future")
            self.assertEqual(virtual["material_sha256"],original["summary"]["material_sha256"])
            self.assertEqual(routed["material_sha256"],virtual["material_sha256"])
            self.assertEqual(routed["reentries"],virtual["reentries"])
            for event in routed["events"]:
                if event["kind"]=="BRUSH_UP_REENTRY":
                    self.assertTrue(event["route_verified_synthetic"])
                    self.assertFalse(event["native_route_verified"])
                    self.assertEqual(event["material_before_sha256"],event["material_after_sha256"])
                    route = event["route"]
                    self.assertTrue(route)
                    self.assertTrue(all(abs(a[0]-b[0])+abs(a[1]-b[1])==1 for a,b in zip(route,route[1:])))
            self.assertEqual(run_arm(recipe,curve,scalar,"routed_future"),routed)

    def test_route_failure_refuses_transfer(self):
        folder = self.source / "moore-reentry"
        recipe, curve, scalar, _ = read_episode(folder)
        # A mask confined to the first 16 points prohibits every future seek;
        # no optimistic transfer through protected terrain may occur.
        allowed = set(curve[:16])
        result = run_arm(recipe,curve,scalar,"routed_future",allowed=allowed)
        self.assertIn(result["terminal"],("NO_VERIFIED_REENTRY","PROTECTED_OR_BLOCKED_PAINT_CELL"))
        self.assertEqual(result["reentries"],0)
        self.assertEqual(result["routed_brush_up_hops"],0)

    def test_stop_and_restart_differ_from_future(self):
        for name in ("hilbert-reentry","moore-reentry","serpentine-reentry"):
            recipe,curve,scalar,_=read_episode(self.source / name)
            future=run_arm(recipe,curve,scalar,"virtual_future")
            stop=run_arm(recipe,curve,scalar,"stop")
            restart=run_arm(recipe,curve,scalar,"tile_restart")
            self.assertLess(stop["strokes"],future["strokes"])
            self.assertEqual(stop["reentries"],0)
            self.assertLessEqual(restart["strokes"],future["strokes"])
            self.assertFalse(any(e.get("kind")=="UNDO_MATERIAL_EDIT" for e in restart["events"]))

    def test_tampering_rehashed_artifact_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = self.source / "moore-reentry"
            for path in source.iterdir():
                if path.is_file():
                    (folder/path.name).write_bytes(path.read_bytes())
            bad = (folder/"guidance.u16le")
            payload = bytearray(bad.read_bytes())
            payload[0] ^= 1
            bad.write_bytes(payload)
            with self.assertRaises(AssertionError):
                read_episode(folder)
            manifest = json.loads((folder/"manifest.json").read_bytes())
            manifest["artifacts"]["guidance.u16le"] = {"bytes":len(payload),"sha256":digest(payload)}
            (folder/"manifest.json").write_bytes(encoded(manifest))
            # Digest rehash is not a signature. This consumer only proves
            # internal custody, not independently reconstructed guidance.
            self.assertEqual(read_episode(folder)[0]["family"],"guided_coverage")

    def test_study_three_matched_cases(self):
        result=study(self.source)
        self.assertEqual(len(result["matched_cases"]),3)
        self.assertFalse(result["native_replay_verified"])
        report={case["case"]:case for case in result["matched_cases"]}
        self.assertEqual(report["moore-reentry"]["arms"]["routed_future"]["components"],4)
        self.assertEqual(report["hilbert-reentry"]["arms"]["routed_future"]["components"],1)


if __name__=="__main__":
    unittest.main()
