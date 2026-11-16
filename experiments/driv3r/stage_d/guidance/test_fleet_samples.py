"""Cheap, deterministic, no-network four-stream sham/guidance/critic fixtures."""
import copy
import hashlib
import json
import struct
import unittest

try:
    from .fleet_samples import (ARMS, VIEWS, M1_COMMIT, M1_SCHEMA_SHA256,
                                build_four_streams, build_stream)
except ImportError:
    from fleet_samples import (ARMS, VIEWS, M1_COMMIT, M1_SCHEMA_SHA256,
                               build_four_streams, build_stream)


def fixture(i=0):
    seed = (101, 203, 307, 409)[i]
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    return {
        "stream_id": f"synthetic.worm.{i}", "seed": seed,
        "source_revision": M1_COMMIT, "snapshot_id": f"synthetic.state.{i}",
        "world_sha256": digest(f"synthetic-world-{seed}"),
        "material_sha256": digest(f"synthetic-material-{seed}"),
        "frame": [10, 20, 2, 2], "guidance_u16": [0, 1000 + i, 33000, 65535],
        "legal_candidates": [
            {"id": "E", "destination_xy": [11, 20], "swept_xy": [[11, 20]], "native_limit_cells": 1},
            {"id": "W", "destination_xy": [10, 20], "swept_xy": [[10, 20]], "native_limit_cells": 1},
            {"id": "STOP", "destination_xy": [10, 20], "swept_xy": [], "native_limit_cells": 0},
        ],
        "host_geometry": {"cursor_xy": [10, 20], "clearance_cells": 3},
        "optional_channels": {
            "signed_preference/v1": [-10, 10, 0, 5],
            "signed_distance/v1": [-5, 0, 4, 8],
            "flow_vector_xy/v1": [[0, 1], [1, 0], [-1, 0], [0, -1]],
        },
        "critic_cases": [
            {"before": {"goal_score": 10, "violations": 0, "progress_cells": 1},
             "after": {"goal_score": 11, "violations": 0, "progress_cells": 2},
             "undo_eligible": True, "hypothetical_receipt_available": True},
            {"before": {"goal_score": 10, "violations": 0, "progress_cells": 1},
             "after": {"goal_score": 8, "violations": 1, "progress_cells": 1},
             "undo_eligible": True, "hypothetical_receipt_available": True},
            {"before": {"goal_score": 10, "violations": 0, "progress_cells": 1},
             "after": {"goal_score": 8, "violations": 1, "progress_cells": 1},
             "undo_eligible": True, "hypothetical_receipt_available": False},
        ]
    }


class FleetSamplesTests(unittest.TestCase):
    def test_exact_m1_pin_and_zero_active_worms(self):
        out = build_four_streams([fixture(i) for i in range(4)])
        self.assertEqual(out["accepted_m1_source"], M1_COMMIT)
        self.assertEqual(out["accepted_m1_schema_sha256"], M1_SCHEMA_SHA256)
        self.assertEqual((out["sample_stream_count"], out["resident_worm_count"],
                          out["native_material_writes"], out["real_undo_executions"],
                          out["model_inference_calls"]), (4, 0, 0, 0, 0))
        without = {k: v for k, v in out.items() if k != "content_sha256_without_self_digest"}
        canonical = json.dumps(without, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        self.assertEqual(out["content_sha256_without_self_digest"],
                         hashlib.sha256(canonical.encode()).hexdigest())

    def test_all_four_arms_same_host_world_and_exact_catalog(self):
        row = build_stream(fixture())
        self.assertEqual(tuple(row["arms"]), ARMS)
        for variant in VIEWS:
            views = [row["arms"][arm]["views"][variant]["model_view"] for arm in ARMS]
            for field in ("snapshot_id", "world_sha256", "material_sha256",
                          "legal_candidate_set_sha256", "legal_candidates", "host_geometry"):
                self.assertEqual(len({json.dumps(v[field], sort_keys=True) for v in views}), 1)
            self.assertEqual(views[0], views[2])
            self.assertEqual(views[1], views[3])
            self.assertNotEqual(views[0], views[1])

    def test_no_critic_leak_or_unscored_action_authority(self):
        row = build_stream(fixture())
        for arm in ARMS:
            for variant in VIEWS:
                model = row["arms"][arm]["views"][variant]["model_view"]
                self.assertNotIn("critic", str(model).lower())
                self.assertNotIn("undo_eligible", model)
                self.assertEqual(set(model["legal_candidates"][0]),
                                 {"id", "destination_xy", "swept_xy", "native_limit_cells"})
                self.assertFalse(row["arms"][arm]["native_material_authority"])
                self.assertFalse(row["arms"][arm]["real_undo_available"])
        self.assertIsNone(row["arms"]["sham"]["post_choice_shadow_critic"])
        self.assertEqual([x["shadow_verdict"] for x in row["arms"]["critic"]["post_choice_shadow_critic"]],
                         ["KEEP", "UNDO", "STOP"])
        self.assertTrue(all(not x["native_undo_executed"] for x in
                            row["arms"]["combined"]["post_choice_shadow_critic"]))

    def test_four_same_state_optional_field_ablations(self):
        row = build_stream(fixture())
        arm = row["arms"]["guidance"]["views"]
        semantics = lambda v: {c["semantic_id"] for c in arm[v]["model_view"]["guidance"]}
        self.assertEqual(semantics("core"), {"positive_scalar/v1"})
        self.assertEqual(semantics("preference"), {"positive_scalar/v1", "signed_preference/v1"})
        self.assertEqual(semantics("geometry"),
                         {"positive_scalar/v1", "signed_distance/v1", "flow_vector_xy/v1"})
        self.assertEqual(len(semantics("all")), 4)
        self.assertEqual(len(arm["core"]["audit_only_ignored"]), 3)
        self.assertNotIn("signed_distance/v1", str(arm["core"]["model_view"]))
        self.assertNotIn("signed_preference/v1", str(arm["core"]["model_view"]))

    def test_exact_u16_signed_and_vector_bytes(self):
        rows = build_stream(fixture())["arms"]["guidance"]["views"]["all"]["model_view"]["guidance"]
        by = {r["semantic_id"]: r for r in rows}
        samples = {
            "positive_scalar/v1": struct.pack("<4H", 0, 1000, 33000, 65535),
            "signed_preference/v1": struct.pack("<4h", -10, 10, 0, 5),
            "signed_distance/v1": struct.pack("<4h", -5, 0, 4, 8),
            "flow_vector_xy/v1": struct.pack("<8h", 0, 1, 1, 0, -1, 0, 0, -1),
        }
        for key, raw in samples.items():
            self.assertEqual(by[key]["sample_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(by[key]["sample_bytes"], len(raw))

    def test_seed_and_source_custody(self):
        rows = build_four_streams([fixture(i) for i in range(4)])["streams"]
        self.assertEqual(len({x["host_source_sha256"] for x in rows}), 4)
        self.assertEqual(len({x["seed"] for x in rows}), 4)
        changed = fixture()
        changed["critic_cases"][0]["after"]["goal_score"] += 1
        self.assertNotEqual(rows[0]["host_source_sha256"], build_stream(changed)["host_source_sha256"])

    def test_input_mutation_cannot_contaminate_output_or_peer_arms(self):
        inp = fixture()
        original = copy.deepcopy(inp)
        row = build_stream(inp)
        self.assertEqual(inp, original)
        row["arms"]["sham"]["views"]["core"]["model_view"]["legal_candidates"][0]["id"] = "TAMPER"
        row["arms"]["sham"]["views"]["all"]["model_view"]["guidance"][0]["values"][0] = 999
        self.assertNotIn("TAMPER", str(row["arms"]["guidance"]["views"]["core"]["model_view"]))
        self.assertEqual(inp, original)

    def test_candidate_presentation_shuffle_does_not_change_identity(self):
        x = fixture()
        a = build_stream(x)
        x["legal_candidates"].reverse()
        b = build_stream(x)
        self.assertEqual(a["legal_candidate_set_sha256"], b["legal_candidate_set_sha256"])
        self.assertEqual(a["arms"]["sham"]["views"]["core"]["view_sha256"],
                         b["arms"]["sham"]["views"]["core"]["view_sha256"])

    def test_reject_guidance_as_permissions(self):
        x = fixture()
        x["legal_candidates"][0]["guidance_score"] = 1
        with self.assertRaisesRegex(ValueError, "shape"):
            build_stream(x)

    def test_reject_unregistered_geometry_score_covert_channel(self):
        x = fixture()
        x["host_geometry"]["critic_instruction"] = "UNDO"
        with self.assertRaisesRegex(ValueError, "hidden scores"):
            build_stream(x)

    def test_reject_unknown_optional_semantic(self):
        x = fixture()
        x["optional_channels"]["unknown/v1"] = [1, 2, 3, 4]
        with self.assertRaisesRegex(ValueError, "M1-registered"):
            build_stream(x)

    def test_reject_signed_overflow_and_bool_values(self):
        for payload in ([32768, 0, 0, 0], [True, 0, 0, 0]):
            x = fixture()
            x["optional_channels"]["signed_preference/v1"] = payload
            with self.assertRaisesRegex(ValueError, "numeric tile"):
                build_stream(x)

    def test_reject_vector_shape_and_tile_overflow(self):
        x = fixture()
        x["optional_channels"]["flow_vector_xy/v1"] = [[0, 0]]
        with self.assertRaisesRegex(ValueError, "signed XY"):
            build_stream(x)
        x = fixture()
        x["frame"] = [10, 20, 128, 64]
        with self.assertRaisesRegex(ValueError, "bounded"):
            build_stream(x)

    def test_reject_duplicate_stream_seeds_and_world_aliases(self):
        inputs = [fixture(i) for i in range(4)]
        inputs[3]["seed"] = inputs[0]["seed"]
        with self.assertRaisesRegex(ValueError, "duplicate seed"):
            build_four_streams(inputs)
        inputs = [fixture(i) for i in range(4)]
        inputs[3]["world_sha256"] = inputs[0]["world_sha256"]
        inputs[3]["material_sha256"] = inputs[0]["material_sha256"]
        with self.assertRaisesRegex(ValueError, "World/material"):
            build_four_streams(inputs)

    def test_reject_world_alias_even_if_material_digest_differs(self):
        # Old pair-only uniqueness check accepted a second nominally
        # independent worm using the exact same World fingerprint.
        inputs = [fixture(i) for i in range(4)]
        inputs[3]["world_sha256"] = inputs[0]["world_sha256"]
        self.assertNotEqual(inputs[3]["material_sha256"], inputs[0]["material_sha256"])
        with self.assertRaisesRegex(ValueError, "source alias: World"):
            build_four_streams(inputs)

    def test_reject_material_alias_even_if_world_digest_differs(self):
        inputs = [fixture(i) for i in range(4)]
        inputs[3]["material_sha256"] = inputs[0]["material_sha256"]
        self.assertNotEqual(inputs[3]["world_sha256"], inputs[0]["world_sha256"])
        with self.assertRaisesRegex(ValueError, "source alias: material"):
            build_four_streams(inputs)

    def test_reject_mixed_source_revisions_with_distinct_worlds_and_seeds(self):
        inputs = [fixture(i) for i in range(4)]
        inputs[3]["source_revision"] = "f" * 40
        with self.assertRaisesRegex(ValueError, "mixed source revisions"):
            build_four_streams(inputs)

    def test_reject_critic_forged_receipt_flag_or_negative_violations(self):
        x = fixture()
        x["critic_cases"][0]["hypothetical_receipt_available"] = "native-verified"
        with self.assertRaisesRegex(ValueError, "shadow flags"):
            build_stream(x)
        x = fixture()
        x["critic_cases"][0]["before"]["violations"] = -1
        with self.assertRaisesRegex(ValueError, "critic metrics"):
            build_stream(x)

    def test_reject_missing_host_legal_catalog_or_all_zero_guide(self):
        x = fixture()
        x["legal_candidates"] = []
        with self.assertRaisesRegex(ValueError, "host candidates"):
            build_stream(x)
        x = fixture()
        x["guidance_u16"] = [0, 0, 0, 0]
        with self.assertRaisesRegex(ValueError, "neutral sham"):
            build_stream(x)


    def test_repaired_fixture_golden_digest_is_source_fenced(self):
        # Independent reconstruction first reproduced the previous published
        # golden byte-for-byte, then recomputed this revised-control digest.
        bundle = build_four_streams([fixture(i) for i in range(4)])
        self.assertEqual(bundle["content_sha256_without_self_digest"],
                         "dfaffcdceb26751e950b217daf64e55120097094fabead06af16c54f9946ba8e")
        self.assertNotEqual(bundle["content_sha256_without_self_digest"],
                            "e81d076d67b94cbf25d70813cfb6eee2f40c5e218c67ad8b20a0ac4973b78b9b")

    def test_all_optional_preference_signals_are_neutral_in_sham(self):
        x = fixture()
        x["optional_channels"]["soft_attraction/v1"] = [0, 100, 65535, 1]
        x["optional_channels"]["soft_avoidance/v1"] = [65535, 3, 0, 99]
        row = build_stream(x)
        for arm in ("sham", "critic"):
            for view in ("preference", "all"):
                channels = row["arms"][arm]["views"][view]["model_view"]["guidance"]
                preferences = [c for c in channels if c["signal_class"] == "preference"]
                self.assertEqual(len(preferences), 4)
                for c in preferences:
                    self.assertTrue(all(v == 0 for v in c["values"]), (arm, view, c["semantic_id"]))
            self.assertEqual(row["arms"][arm]["views"]["core"]["model_view"]["guidance"][0]["values"],
                             [0, 0, 0, 0])
        active = row["arms"]["guidance"]["views"]["all"]["model_view"]["guidance"]
        self.assertTrue(all(any(v != 0 for v in c["values"])
                            for c in active if c["signal_class"] == "preference"))

    def test_sham_still_exposes_same_factual_geometry_metrics(self):
        row = build_stream(fixture())
        for view in ("geometry", "all"):
            samples = {}
            for arm in ARMS:
                channels = row["arms"][arm]["views"][view]["model_view"]["guidance"]
                samples[arm] = {c["semantic_id"]: (c["sample_sha256"], c["values"])
                                for c in channels if c["signal_class"] == "geometry_metric"}
            self.assertTrue(samples["sham"])
            self.assertEqual(samples["sham"], samples["guidance"])
            self.assertEqual(samples["sham"], samples["critic"])
            self.assertEqual(samples["sham"], samples["combined"])

    def test_sham_preference_hash_is_of_neutral_samples_not_active_source(self):
        x = fixture()
        row = build_stream(x)
        neutral_raw = struct.pack("<4h", 0, 0, 0, 0)
        active_raw = struct.pack("<4h", *x["optional_channels"]["signed_preference/v1"])
        for arm in ("sham", "critic"):
            by_id = {c["semantic_id"]: c for c in
                     row["arms"][arm]["views"]["all"]["model_view"]["guidance"]}
            pref = by_id["signed_preference/v1"]
            self.assertEqual(pref["sample_sha256"], hashlib.sha256(neutral_raw).hexdigest())
            self.assertNotEqual(pref["sample_sha256"], hashlib.sha256(active_raw).hexdigest())
        guide = {c["semantic_id"]: c for c in
                 row["arms"]["guidance"]["views"]["all"]["model_view"]["guidance"]}
        self.assertEqual(guide["signed_preference/v1"]["sample_sha256"],
                         hashlib.sha256(active_raw).hexdigest())

    def test_all_view_ablation_diff_is_only_preference_signals(self):
        row = build_stream(fixture())
        for variant in VIEWS:
            x = row["arms"]["sham"]["views"][variant]["model_view"]
            y = row["arms"]["guidance"]["views"][variant]["model_view"]
            self.assertEqual(set(x), set(y))
            self.assertEqual({k: v for k, v in x.items() if k != "guidance"},
                             {k: v for k, v in y.items() if k != "guidance"})
            left = {c["semantic_id"]: c for c in x["guidance"]}
            right = {c["semantic_id"]: c for c in y["guidance"]}
            self.assertEqual(set(left), set(right))
            for semantic in left:
                self.assertEqual(left[semantic]["encoding"], right[semantic]["encoding"])
                if left[semantic]["signal_class"] == "geometry_metric":
                    self.assertEqual(left[semantic], right[semantic])
                else:
                    self.assertTrue(all(v == 0 for v in left[semantic]["values"]))


if __name__ == "__main__":
    unittest.main()
