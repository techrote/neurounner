"""Cheap deterministic-only tests; deliberately not a native/held-out evaluation."""
import unittest

from experiments.driv3r.stage_d.direct import offline as d


class FixtureTests(unittest.TestCase):
    def setUp(self):
        self.cases = d.fixtures()
        self.dev = [f for f in self.cases if f.split == "development"]
        self.sealed = [f for f in self.cases if f.split == "heldout"]

    def test_structural_split_frozen_and_disjoint(self):
        self.assertEqual(len(self.dev), 7)
        self.assertEqual(len(self.sealed), 4)
        self.assertEqual(len({f.family for f in self.cases}), 11)
        self.assertEqual(len({f.sha256 for f in self.cases}), len(self.cases))
        self.assertTrue(all(f.budget == 36 for f in self.cases))

    def test_fixture_topology_and_target(self):
        for f in self.cases:  # topology only, no model outcomes on heldout
            with self.subTest(fixture=f.fixture_id):
                d.check_fixture(f)
                self.assertTrue(f.legal(f.start))
                self.assertTrue(f.legal(f.target))
                self.assertNotIn(f.start,f.forbidden)
                self.assertNotIn(f.target,f.forbidden)

    def test_wall_identity_not_distractor(self):
        for f in self.dev:
            if f.task == "wall_offset":
                self.assertEqual(f.offset,8)
                self.assertEqual(f.wall_profile[0]+8,f.start[0])
                self.assertNotIn((34,10),f.selected_wall)
                self.assertEqual(d.wall_distance(f,(34,10)),
                                 min(d.hypot(34-x,10-y) for x,y in f.selected_wall))

    def test_segment_never_crosses_blocked_cell(self):
        f = d.routing_fixture("gate","development")
        opts = d.candidates(f,(10,8),(1,2,4,8))
        self.assertFalse(any(c.candidate_id.startswith("E") for c in opts))
        self.assertTrue(all(all(f.legal(p) for p in c.path) for c in opts))

    def test_retrace_count(self):
        f = self.dev[0]
        a,b = f.start,(f.start[0]+1,f.start[1])
        score=d.score_episode(f,[a,b,a,b],3,"policy_stop")
        self.assertEqual(score["retraced_cells"],2)
        self.assertTrue(score["false_stop"])

    def test_deterministic_random_and_hash_custody(self):
        f = d.routing_fixture("gate","development")
        a=d.run_episode(f,"random_legal",seed=13)
        b=d.run_episode(f,"random_legal",seed=13)
        self.assertEqual(a,b)
        self.assertEqual(a["fixture_sha256"],f.sha256)
        self.assertFalse(a["native_qualified"])
        self.assertTrue(all(e["native_receipt_ref"] is None for e in a["events"]))
        self.assertEqual(a["metrics"]["native_receipts"],0)

    def test_astar_dev_arrives_under_budget(self):
        # Only development tasks: held-out policy outcomes remain unexamined.
        for f in self.dev:
            with self.subTest(fixture=f.fixture_id):
                policy="contour_astar" if f.task=="wall_offset" else "astar"
                r=d.run_episode(f,policy)
                self.assertTrue(r["metrics"]["arrived"],r["fixture"])
                self.assertLessEqual(r["metrics"]["attempted_decisions"],36)
                self.assertEqual(r["metrics"]["collisions"],0)
                self.assertEqual(r["metrics"]["material_z_violations"],0)
                self.assertEqual(r["metrics"]["protected_violations"],0)
                self.assertEqual(r["metrics"]["native_readback"],"not_performed")
                self.assertTrue(all(e["observation"]["native_world_ref"] is None
                                    for e in r["events"]))
                self.assertEqual(len(r["events"]),r["metrics"]["attempted_decisions"])
                for e in r["events"]:
                    choices=e["observation"]["candidates"]
                    self.assertIn(e["choice_id"],{c["candidate_id"] for c in choices})
                    self.assertEqual(e["observation"]["candidate_set_sha256"],d.digest(choices))

    def test_stop_stays_nonmutating(self):
        f=d.routing_fixture("gate","development")
        opts=d.candidates(f,f.start)
        self.assertEqual(opts[0].candidate_id,"STOP")
        self.assertEqual(opts[0].path,())
        self.assertEqual(d.score_episode(f,[f.start],1,"policy_stop")["native_edit_operations"],0)

    def test_macro_swept_and_offline_only(self):
        f=d.routing_fixture("slalom","development")
        r=d.run_episode(f,"astar",lengths=(1,2,4,8))
        for e in r["events"]:
            self.assertLessEqual(len(next(c["path"] for c in
                e["observation"]["candidates"] if c["candidate_id"]==e["choice_id"])),8)
        self.assertEqual(r["sequential_native_receipts"],0)


if __name__ == "__main__":
    unittest.main()
